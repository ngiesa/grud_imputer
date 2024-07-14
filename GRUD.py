"""
Created on Sat May 12 16:48:54 2018

@author: Zhiyong
"""
import os

import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import numpy as np
import pandas as pd
import time


class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()  # delta calculation filter linear layer? why?
        self.in_features = in_features
        self.out_features = out_features

        # TODO here the decays are just set if the actual value is absent

        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    #         print(self.weight.data)
    #         print(self.bias.data)

    def forward(self, input):
        #         print(self.filter_square_matrix.mul(self.weight))
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'


class GRUD(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size, X_mean, output_last=False):
        """
        Recurrent Neural Networks for Multivariate Times Series with Missing Values
        GRU-D: GRU exploit two representations of informative missingness patterns, i.e., masking and time interval.
        cell_size is the size of cell_state.
        
        Implemented based on the paper: 
        @article{che2018recurrent,
          title={Recurrent neural networks for multivariate time series with missing values},
          author={Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan},
          journal={Scientific reports},
          volume={8},
          number={1},
          pages={6085},
          year={2018},
          publisher={Nature Publishing Group}
        }
        
        GRU-D:
            input_size: variable dimension of each time
            hidden_size: dimension of hidden_state
            mask_size: dimension of masking vector
            X_mean: the mean of the historical input data
        """

        super(GRUD, self).__init__()

        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.identity = torch.eye(input_size).cuda()
            self.zeros = Variable(torch.zeros(input_size).cuda())
            self.X_mean = Variable(torch.Tensor(X_mean).cuda())
        else:
            self.identity = torch.eye(input_size)
            self.zeros = Variable(torch.zeros(input_size))
            self.X_mean = Variable(torch.Tensor(X_mean))

        # TODO three linear layers?
        self.zl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)  # update gate
        self.rl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)  # reset gate
        self.hl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)  # hidden layer

        self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, self.identity)

        self.gamma_h_l = nn.Linear(self.delta_size, self.delta_size)

        self.output_last = output_last

    def step(self, x, x_last_obsv, x_mean, h, mask, delta):

        input_x = []
        mean_x = []
        deltas_x = []
        deltas_h = []

        delta_x = torch.exp(-torch.max(self.zeros, self.gamma_x_l(delta)))  # TODO Figure 3 in GRU-D paper
        delta_h = torch.exp(-torch.max(self.zeros, self.gamma_h_l(delta)))

        # print(f"{delta_x.detach().numpy().shape = }")
        # print(f"{delta_h.detach().numpy().shape = }")
        # print(f"{x_mean = }"

        deltas_x.append(pd.DataFrame(delta_x.detach().numpy()))
        pd.concat(deltas_x).to_csv("decay_x.csv", index=False)

        deltas_h.append(pd.DataFrame(delta_h.detach().numpy()))
        pd.concat(deltas_h).to_csv("decay_h.csv", index=False)

        input_x.append(pd.DataFrame(x.detach().numpy()))
        pd.concat(input_x).to_csv("input_x.csv", index=False)

        mean_x.append(pd.DataFrame(x_mean.detach().numpy()))
        pd.concat(mean_x).to_csv("mean_x.csv", index=False)

        # print(delta_x)  # TODO store deltas and describe more

        # print(f"{delta_x.shape = }, {delta_h.shape = }")

        x = mask * x + (1 - mask) * (
                delta_x * x_last_obsv + (1 - delta_x) * x_mean)  # TODO equation 5 in paper delta x = decay?

        # TODO (1 - delta_x) * x_mean) tendency to the mean -> if delta x is small, then we have a big tendency
        #  towards mean and vice versa

        h = delta_h * h
        # print(f"{h.shape = }, {mask.shape = }, {x.shape = }")

        combined = torch.cat((x, h, mask), 1)
        z = F.sigmoid(self.zl(combined))
        r = F.sigmoid(self.rl(combined))
        combined_r = torch.cat((x, r * h, mask), 1)
        h_tilde = F.tanh(self.hl(combined_r))  # TODO
        h = (1 - z) * h + z * h_tilde

        return h

    def forward(self, input):
        # print(f"{input.shape = }")
        batch_size = input.size(0)
        type_size = input.size(1)
        step_size = input.size(2)
        spatial_size = input.size(3)

        Hidden_State = self.initHidden(batch_size)
        X = torch.squeeze(input[:, 0, :, :])
        X_last_obsv = torch.squeeze(input[:, 1, :, :])
        Mask = torch.squeeze(input[:, 2, :, :])  # TODO at index 2 maks is stored
        Delta = torch.squeeze(input[:, 3, :, :])  # TODO at index 3 the delta is stored
        # print(f"{X.shape = }")
        # print(f"{X_last_obsv.shape = }")
        # print(f"{Mask.shape = }")
        # print(f"{Delta.shape = }")

        outputs = None
        for i in range(step_size):
            # print(X.shape)
            a = torch.squeeze(X[:, i:i + 1, :])  # X[:, i:i + 1] # TODO model inputs all previous sequences
            b = torch.squeeze(
                X_last_obsv[:, i:i + 1, :])  # X_last_obsv[:, i:i + 1] # TODO model inputs just the last input ??
            c = torch.squeeze(self.X_mean[:, i:i + 1, :])

            d = torch.squeeze(Mask[:, i:i + 1, :])  # Mask[:, i:i + 1]
            e = torch.squeeze(Delta[:, i:i + 1, :])  # Delta[:, i:i + 1]
            # print(f"{a.shape = }")
            # print(f"{b.shape = }")
            # print(f"{c.shape = }")
            # print(f"{d.shape = }")
            # print(f"{e.shape = }")
            # print(f"{a.shape = }, {b.shape = }, {c.shape = }, {d.shape = }, {e.shape = }")
            Hidden_State = self.step(a
                                     , b
                                     , c
                                     , Hidden_State
                                     , d
                                     , e)
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)

        if self.output_last:
            return outputs[:, -1, :]
        else:
            return outputs

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State
