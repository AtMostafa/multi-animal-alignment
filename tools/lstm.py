import numpy as np
import pandas as pd
import pyaldata as pyal
from sklearn.decomposition import PCA
from typing import Callable
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)



class LstmDecoder(nn.Module):
    """
    LSTM decoder for timeseries
    based on: https://towardsdatascience.com/pytorch-lstms-for-time-series-data-cd16190929d7
    """
    def __init__(self, input_size=10, hidden_dim=128, tagset_size=2):
        super().__init__()
        self.hidden_layers = hidden_dim
        self.input_size = input_size
        self.lstm1 = nn.LSTMCell(input_size, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, tagset_size)

    def forward(self, signal):
        """
        The `forward` method
        `signal`: of size time x trial x features
        """
        outputs, n_samples = [], signal.size(1)
        h_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        
        for input_t in signal.split(1, dim=0):
            # N, 1
            h_t, c_t = self.lstm1(torch.squeeze(input_t), (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            output = self.linear(h_t2) # output from the last FC layer
            outputs.append(output)

        # transform list to tensor    
        outputs = torch.cat(outputs, dim=1)
        return outputs
