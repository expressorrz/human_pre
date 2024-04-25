import torch
import copy
import numpy as np
import torch.nn as nn
import math


class LSTM_model(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size_fc, num_layers, output_size):
        super(LSTM_model, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        self.bilstm = torch.nn.LSTM(input_size, hidden_size1, num_layers, batch_first=True, bidirectional=True)
        self.lstm = torch.nn.LSTM(hidden_size1*2, hidden_size2, num_layers, batch_first=True)

        self.fc1 = torch.nn.Linear(hidden_size2, hidden_size_fc)
        self.fc2 = torch.nn.Linear(hidden_size_fc, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        h0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size1).to(x.device)
        c0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size1).to(x.device)

        h1 = torch.randn(self.num_layers, x.size(0), self.hidden_size2).to(x.device)
        c1 = torch.randn(self.num_layers, x.size(0), self.hidden_size2).to(x.device)

        out, _ = self.bilstm(x, (h0, c0))
        out, _ = self.lstm(out, (h1, c1))

        out = self.fc1(out)
        out = self.fc2(out)
        out = out.permute(0, 2, 1)
        return out
    