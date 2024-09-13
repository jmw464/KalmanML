#!/usr/bin/env python

import torch.nn as nn
import torch.nn.functional as F


class PolicyNN(nn.Module):
    def __init__(self, ninputs, modeldim, nlayers, dropout):
        super().__init__()

        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(nn.Linear(ninputs, modeldim))
        for i in range(nlayers):
            self.linear_layers.append(nn.BatchNorm1d(modeldim))
            self.linear_layers.append(nn.Dropout(p=dropout))
            self.linear_layers.append(nn.ReLU())
            self.linear_layers.append(nn.Linear(modeldim, modeldim))
        self.linear_layers.append(nn.Linear(modeldim, 1))
        self.linear_layers.append(nn.Sigmoid())

    def forward(self, x):
        h = self.linear_layers(x)
        return h