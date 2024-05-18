import torch
import copy
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F

class ST_Transformer(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size, nhead=4, dropout=0.0, noise_level=0.01):
        super(ST_Transformer, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.spatial_embedding = spatial_embedding(hidden_dim)


        self.temporal_embedding = nn.Linear(12, 12)

        self.embedding = nn.Linear(75, hidden_dim)

        self.GraphAttentionLayer = GraphAttentionLayer(input_size, hidden_dim, dropout, 0.2, concat=True)

        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)


        # spatial transformer encoder
        st_encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        encode_norm = nn.LayerNorm(hidden_dim)
        self.s_transformer_encoder = nn.TransformerEncoder(st_encoder_layers, num_layers=num_layers, norm=encode_norm)

        # temporal transformer encoder
        tt_encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        encode_norm = nn.LayerNorm(hidden_dim)
        self.t_transformer_encoder = nn.TransformerEncoder(tt_encoder_layers, num_layers=num_layers, norm=encode_norm)

        # transformer decoder
        transformer_decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim*2, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        decode_norm = nn.LayerNorm(hidden_dim*2)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layers, num_layers=num_layers, norm=decode_norm)


        self.fc = nn.Linear(hidden_dim*2, output_size)


    
    def forward(self, x, adj):
        x0 = x                                                              # batchsize, seq_len, 75
        
        x_s = x                                                             # batchsize, seq_len, 75
        x_t = x.transpose(1, 2)                                             # batchsize, seq_len, 75

        # spatial embedding
        a = self.spatial_embedding(x_s, adj)
        b = self.embedding(x_s) 
        print('a shape:', a.size())
        print('b shape:', b.size())
        x_s = self.spatial_embedding(x_s, adj) + self.embedding(x_s)        # batchsize, seq_len, 64

        
        # temporal embedding
        x_t = self.temporal_embedding(x_t).transpose(1, 2) + x0             # batchsize, 75, seq_len
        x_t = self.embedding(x_t)                                           # batchsize, seq_len, 64

        x0 = torch.cat((x_s, x_t), dim=2)                                   # batchsize, seq_len, 128

        # spatial transformer
        x_s = self.s_transformer_encoder(x_s)
        # print('x_s shape:', x_s.size())


        
        # temporal transformer
        x_t = self.t_transformer_encoder(x_t)
        # print('x_t shape:', x_t.size())

        out = torch.cat((x_s, x_t), dim=2)
        out = self.transformer_decoder(out, x0)
        out = self.fc(out)

        out = out[:, -6:, :]
        return out
    
class spatial_embedding(nn.Module):
    def __init__(self, hidden_dim):
        super(spatial_embedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Parameter(torch.empty(size=(3, self.hidden_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * self.hidden_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(0.0)

        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x, adj):
        # x: batchsize, seq_len, 75
        x = x.reshape(x.size(0), x.size(1), 25, 3)                           # batchsize, seq_len, 25, 3
        Wh = torch.matmul(x, self.W).to(x.device)                              # batchsize, seq_len, 25, 64

        e_matrix = torch.zeros(x.size(0), x.size(1), 25, 25).to(x.device)
        a_matrix = torch.zeros(x.size(0), x.size(1), 25, 25).to(x.device)

        Wh1 = torch.matmul(Wh, self.a[:self.hidden_dim, :])
        Wh2 = torch.matmul(Wh, self.a[:self.hidden_dim, :])

        for i in range(25):
            for j in range(25):
                if adj[i, j] == 1:
                    a = self.leaky_relu(Wh1[:, :, i, :] + Wh2[:, :, j, :]).squeeze(2)
                    e_matrix[:, :, i, j] = a

            
            a_matrix[:, :, i, :] = F.softmax(e_matrix[:, :, i, :])
            a_matrix[:, :, i, :] = self.dropout(a_matrix[:, :, i, :])


        h_prime = torch.matmul(a_matrix[:, :, :, :],  Wh[:, :, :, :])

        return h_prime
    
class temporal_embedding(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(temporal_embedding, self).__init__()
        self.temporal_embedding = nn.Linear(input_size, hidden_dim)
    
    def forward(self, x):
        x = self.temporal_embedding(x)
        return x
    
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # (N, in_features) * (in_features, out_features) -> (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size, nhead=1, dropout=0.0, noise_level=0.01):
        super(Transformer, self).__init__()

        self.num_layers = num_layers

        self.embeding = nn.Linear(12, hidden_dim)

        # encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        encode_norm = nn.LayerNorm(hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers, norm=encode_norm)

        # decoder
        decode_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        decode_norm = nn.LayerNorm(hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(decode_layer, num_layers=num_layers, norm=decode_norm)

        # Fc
        self.fc = nn.Linear(hidden_dim, 6)
    
    def forward(self, x):
        x = self.embeding(x)

        # encoder
        out = self.transformer_encoder(x)

        # decoder
        features = self.transformer_decoder(out, x)


        # Fc
        out = self.fc(features)[:,-6:,:]

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