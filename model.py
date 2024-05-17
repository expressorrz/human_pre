import torch
import copy
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F

class S2Transformer(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size, nhead=4, dropout=0.0, noise_level=0.01):
        super(S2Transformer, self).__init__()

        self.num_layers = num_layers

        # positional encoding
        self.embeding = nn.Linear(input_size, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim=hidden_dim)

        # encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        encode_norm = nn.LayerNorm(hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers, norm=encode_norm)

        # decoder
        decode_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        decode_norm = nn.LayerNorm(hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(decode_layer, num_layers=num_layers, norm=decode_norm)

        # Fc
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        x = self.embeding(x)
        x = self.pos(x)         # batch size, seq_len, input_size

        # encoder
        out = self.transformer_encoder(x)

        # decoder
        features = self.transformer_decoder(out, x)

        # Fc
        out = self.fc(features)[:, -6:, :]


        return out

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size, nhead=1, dropout=0.0, noise_level=0.01):
        super(Transformer, self).__init__()

        self.num_layers = num_layers

        self.embeding = nn.Linear(input_size, hidden_dim)

        # encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        encode_norm = nn.LayerNorm(hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers, norm=encode_norm)

        # decoder
        decode_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        decode_norm = nn.LayerNorm(hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(decode_layer, num_layers=num_layers, norm=decode_norm)

        # Fc
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        x = self.embeding(x)

        # encoder
        out = self.transformer_encoder(x)

        # decoder
        features = self.transformer_decoder(out, x)


        # Fc
        out = self.fc(features)[:, -6:, :]

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)

        pe = torch.zeros(max_len, hidden_dim)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))

        a = position * div_term

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        return x


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
        self.fc2 = nn.Linear(hidden_size1, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out[:, -6:, :]
    


class CNN_model(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_size_fc, output_size):
        super(CNN_model, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(hidden_channels, hidden_size_fc)
        self.fc2 = nn.Linear(hidden_size_fc, output_size)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.transpose(1, 2)[:,-6:,:]

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

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
        out = self.fc1(out[:, -6:, :])  # Use the last output of GRU
        out = self.fc2(out)
        return out