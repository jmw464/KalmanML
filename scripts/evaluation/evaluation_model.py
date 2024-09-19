#!/usr/bin/env python

import math
import numpy as np
import torch as th
import torch.nn as nn

class TracksDataset(th.utils.data.Dataset):
    def __init__(self, inputs, labels, dataset_mask):
        self.inputs = inputs
        self.labels = th.tensor(labels)
        self.dataset_mask = th.tensor(dataset_mask)
        self.ntracks, self.nhits = labels.shape
        self.ninputs = len(list(inputs.dtype.names))

    def __len__(self):
        return self.dataset_mask.sum()

    def __getitem__(self, idx):
        idx = th.nonzero(self.dataset_mask).flatten()[idx] # get actual index of non-masked element
        element = self.inputs[idx]
        inputs = th.tensor(np.vstack([element[field].astype(np.double) for field in element.dtype.names]).transpose())
        labels = th.unsqueeze(self.labels[idx], -1).double()

        return th.nan_to_num(inputs, nan=0.0), labels


class PositionalEncoding(nn.Module):
    def __init__(self, modeldim, dropout, max_len):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)

        # Encoding - From formula
        pos_encoding = th.zeros(max_len, modeldim)
        positions_list = th.arange(0, max_len, dtype=th.float).view(-1, 1)
        division_term = th.exp(th.arange(0, modeldim, 2).float()*(-math.log(10000.0))/modeldim)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = th.sin(positions_list*division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = th.cos(positions_list*division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)

    def forward(self, token_embedding):
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class EvalNN(nn.Module):
    def __init__(self, ninputs, modeldim, nheads, nlayers, encode_pos, dropout, max_len):
        super().__init__()

        self.linear_in = nn.Linear(ninputs, modeldim)
        self.activation_in = nn.ReLU()
        self.linear_out = nn.Linear(modeldim, 1)
        self.activation_out = nn.Sigmoid()
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=modeldim, nhead=nheads), num_layers=nlayers)
        
        self.encode_pos = encode_pos
        self.positional_encoder = PositionalEncoding(modeldim=modeldim, dropout=dropout, max_len=max_len)

    def forward(self, h):
        h = self.activation_in(self.linear_in(h))
        if self.encode_pos:
            h = self.positional_encoder(h)
        h = self.transformer(h)
        h = self.activation_out(self.linear_out(h))
        #h = th.transpose(h,1,2)
        return h


