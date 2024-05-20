import torch
import copy
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F


class ST_Transformer(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size, nhead, dropout=0.0):
        super(ST_Transformer, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.spatial_embedding = SpatialAttention(hidden_dim=hidden_dim)


        self.temporal_embedding = TemporalAttention(seq_len=12, hidden_dim=hidden_dim)

        self.embedding = nn.Linear(75, hidden_dim)

        # spatial transformer encoder
        st_encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        st_encode_norm = nn.LayerNorm(hidden_dim)
        self.s_transformer_encoder = nn.TransformerEncoder(st_encoder_layers, num_layers=num_layers, norm=st_encode_norm)

        # temporal transformer encoder
        tt_encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        tt_encode_norm = nn.LayerNorm(hidden_dim)
        self.t_transformer_encoder = nn.TransformerEncoder(tt_encoder_layers, num_layers=num_layers, norm=tt_encode_norm)

        # transformer decoder
        transformer_decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim*2, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        decode_norm = nn.LayerNorm(hidden_dim*2)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layers, num_layers=num_layers, norm=decode_norm)

        # GRU
        self.gru_decoder = nn.GRU(hidden_dim*2, self.hidden_dim*2, 2, batch_first=True)

        # fc_layer
        self.fc1_layer = nn.Linear(hidden_dim*2, 75)
    
    def forward(self, x, adj):
        x0 = x                                                              # batchsize, seq_len, 75
        
        x_s = x                                                             # batchsize, seq_len, 75
        x_t = x.transpose(1, 2)                                             # batchsize, 75, seq_len

        """
            Spatial and Temporal Attention
        """

        # spatial embedding
        x_s = self.spatial_embedding(x_s, adj)                              # batchsize, seq_len, 75
        x_s = x_s + x0
        x_s = self.embedding(x_s)                                           # batchsize, seq_len, hidden_dim

        
        # temporal embedding
        x_t = self.temporal_embedding(x_t).transpose(1, 2)             # batchsize, seq_len, 75
        x_t = x_t + x0
        x_t = self.embedding(x_t)                                           # batchsize, seq_len, hidden_dim

        x0 = torch.cat((x_s, x_t), dim=2)                                   # batchsize, seq_len, hidden_dim * 2

        """
            Spatial and Temporal Transformer Encoder
        """
        # spatial transformer
        x_s1 = self.s_transformer_encoder(x_s)                               # batchsize, seq_len, hidden_dim

        # temporal transformer
        x_t1 = self.t_transformer_encoder(x_t)                               # batchsize, seq_len, hidden_dim

        out = torch.cat((x_s1, x_t1), dim=2)

        # """
        #     Transformer Decoder
        # """
        # out = self.transformer_decoder(out, x0)                             # batchsize, seq_len, hidden_dim * 2

        """
            GRU decoder layer
        """
        h3 = torch.randn(2, x.size(0), self.hidden_dim*2).to(x.device)

        out, _ = self.gru_decoder(out, h3)

        """
            fc layer
        """
        out = self.fc1_layer(out)                                        # batchsize, seq_len, hidden_dim
        out = out[:, -6:, :]

        return out


class TemporalAttention(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.query_linear = nn.Linear(seq_len, hidden_dim)
        self.key_linear = nn.Linear(seq_len, hidden_dim)
        self.value_linear = nn.Linear(seq_len, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, seq_len)

    def forward(self, x):
        
        # Temporal attention mechanism
        Q_t = self.query_linear(x)                                                                  # (batch_size, 75, hidden_dim)
        K_t = self.key_linear(x)                                                                    # (batch_size, 75, hidden_dim)
        V_t = self.value_linear(x)                                                                  # (batch_size, 75, hidden_dim)

        # Compute temporal attention scores
        temporal_attention_scores = torch.bmm(Q_t, K_t.transpose(1, 2)) / (self.hidden_dim ** 0.5)  # (batch_size, 75, 75)
        temporal_attention_weights = F.softmax(temporal_attention_scores, dim=-1)                   # (batch_size, 75, 75)

        # Compute temporal attention output
        temporal_attention_output = torch.bmm(temporal_attention_weights, V_t)                      # (batch_size, 75, hidden_dim)

        # Final updated features
        final_updated_features = self.out_linear(temporal_attention_output)                         # (batch_size, 75, seq_len)
        
        return final_updated_features


class SpatialAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SpatialAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Parameter(torch.empty(size=(1, self.hidden_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * self.hidden_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(0.0)

        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x, adj):
        adj = torch.tensor(adj, dtype=torch.float32, device=x.device)

        # Adjust dimensions and apply linear transformation
        x = x.unsqueeze(3)  # batchsize, seq_len, 75, 1
        Wh = torch.matmul(x, self.W)  # batchsize, seq_len, 75, hidden_dim

        # Compute attention scores using broadcasting
        Wh1 = torch.matmul(Wh, self.a[:self.hidden_dim]).transpose(2, 3)  # batchsize, seq_len, 1, 75
        Wh2 = torch.matmul(Wh, self.a[self.hidden_dim:]).transpose(2, 3)  # batchsize, seq_len, 1, 75
        
        e = self.leaky_relu(Wh1 + Wh2.transpose(2, 3))  # batchsize, seq_len, 75, 75

        # Apply mask to e
        e = e.masked_fill(adj.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

        # Apply softmax and dropout
        a = F.softmax(e, dim=-1)
        a = self.dropout(a)
        
        # Compute final output
        h_prime = torch.matmul(a, Wh)  # batchsize, seq_len, 75, hidden_dim
        h_prime = h_prime.mean(dim=-1)  # batchsize, seq_len, 75

        return h_prime



class Transformer(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size, nhead, dropout=0.0):
        super(Transformer, self).__init__()

        self.num_layers = num_layers

        self.embeding = nn.Linear(75, hidden_dim)

        # encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        encode_norm = nn.LayerNorm(hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers, norm=encode_norm)

        # decoder
        decode_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        decode_norm = nn.LayerNorm(hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(decode_layer, num_layers=num_layers, norm=decode_norm)

        # Fc
        self.fc = nn.Linear(hidden_dim, 75)
    
    def forward(self, x):
        x = self.embeding(x)

        # encoder
        out = self.transformer_encoder(x)

        # decoder
        features = self.transformer_decoder(out, x)


        # Fc
        out = self.fc(features[:,-6:,:])

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
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_model, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -6:, :])

        return out
    
class BiLSTM_model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM_model, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.bilstm = torch.nn.LSTM(input_size, hidden_size, self.num_layers, batch_first=True, bidirectional=True)

        self.fc = torch.nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)


        out, _ = self.bilstm(x, (h0, c0))

        out = self.fc(out[:, -6:, :])
        return out


class MLP_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)



    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out[:, -6:, :]
    


class CNN_model(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_size_fc, output_size):
        super(CNN_model, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        
        self.fc = nn.Linear(hidden_channels, output_size)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.transpose(1, 2)[:,-6:,:]

        x = self.fc(x)
        
        return x

class GRU_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU_model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = 64

        self.gru = nn.GRU(input_size, self.hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -6:, :])  # Use the last output of GRU
        return out