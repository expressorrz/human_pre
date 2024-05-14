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
        h0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size1).to(x.device)
        c0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size1).to(x.device)

        h1 = torch.randn(self.num_layers, x.size(0), self.hidden_size2).to(x.device)
        c1 = torch.randn(self.num_layers, x.size(0), self.hidden_size2).to(x.device)

        out, _ = self.bilstm(x, (h0, c0))
        out, _ = self.lstm(out, (h1, c1))

        out = self.fc1(out[:, -6:, :])
        out = self.fc2(out)
        return out


class MLP_model(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP_model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out[:, -6:, :]
    


class CNN_model(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNN_model, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * (input_size // 4), 100)  # Adjust input_size accordingly
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)  # Flatten the output for the fully connected layer
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        return out

class GRU_model(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size_fc, num_layers, output_size):
        super(GRU_model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size1 = hidden_size1

        self.gru = nn.GRU(input_size, hidden_size1, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size1, hidden_size_fc)
        self.fc2 = nn.Linear(hidden_size_fc, output_size)

    def forward(self, x):
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size1).to(x.device)

        out, _ = self.gru(x, h0)
        out = self.fc1(out[:, -1, :])  # Use the last output of GRU
        out = self.fc2(out)
        return out