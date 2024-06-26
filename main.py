#! /home/expresso/software/anaconda3/envs/spinningup/bin/python
import os
import numpy as np
import torch
import time
import tqdm
import pandas as pd
from param import configs
from utils import *
from model import *


str_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def main():
    # device
    device = torch.device(configs.device)

    # setup seed
    setup_seed(configs.seed)

    # load dateset
    train_loader, vali_loader, test_loader, scaler = load_dataset()

    # edge connection
    edge_set = [[13, 14], [14, 15], [12, 13], [0, 12], [0, 16], [16, 17], [17, 18], [18, 19], [0, 1], [1, 20], [20, 2], [2, 3], [20, 8], [8, 9], [9, 10], [10, 11], [20, 4], [4, 5], [5, 6], [6, 7], [10, 24], [11, 23], [6, 22], [7, 21]]

    adj_matrix = np.eye(25)
    for u, v in edge_set:
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1
    
    # update adj
    temp_adj = np.zeros((75, 75), dtype=int)
    for i in range(25):
        for j in range(25):
            if adj_matrix[i, j] == 1:
                for k in range(3):
                    for l in range(3):
                        temp_adj[i*3 + k, j*3 + l] = 1

    adj_matrix = temp_adj

    print('Model Name:', configs.model_name)

    # model selection
    if configs.model_name == 'MLP':
        model = MLP_model(configs.input_dim, configs.hidden_dim, configs.output_dim).to(device)
    elif configs.model_name == 'CNN':
        model = CNN_model(configs.input_dim, configs.hidden_dim, configs.hidden_dim_fc, configs.output_dim).to(device)
    elif configs.model_name == 'GRU':
        model = GRU_model(configs.input_dim, configs.hidden_dim, configs.num_layers, configs.output_dim).to(device)
    elif configs.model_name == 'LSTM':
        model = LSTM_model(configs.input_dim, configs.hidden_dim, configs.num_layers, configs.output_dim).to(device)
    elif configs.model_name == 'BiLSTM':
        model = BiLSTM_model(configs.input_dim, configs.hidden_dim, configs.num_layers, configs.output_dim).to(device)
    elif configs.model_name == 'Transformer':
        model = Transformer(input_size=configs.input_dim, hidden_dim=configs.hidden_dim_trans, num_layers=configs.num_layers, output_size=configs.output_dim, nhead=configs.num_heads).to(device)
    elif configs.model_name == 'ST_Transformer':
        model = ST_Transformer(input_size=configs.input_dim, hidden_dim=configs.hidden_dim_trans, num_layers=configs.num_layers, output_size=configs.output_dim, nhead=configs.num_heads).to(device)

    # loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

    # log
    training_log = []
    validation_log = []

    # train
    for epoch in range(configs.epoch):
        # train
        model.train()
        train_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            if configs.model_name == 'ST_Transformer':
                output = model(x, adj_matrix)
            else:
                output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        print('Epoch:', epoch, 'Train Loss:', train_loss)
        training_log.append(train_loss)

        if epoch % 10 == 0:
            # validation
            model.eval()
            vali_loss = 0
            with torch.no_grad():
                for i, (x, y) in enumerate(vali_loader):
                    x, y = x.to(device), y.to(device)
                    if configs.model_name == 'ST_Transformer':
                        output = model(x, adj_matrix)
                    else:
                        output = model(x)
                    loss = criterion(output, y)
                    vali_loss += loss.item()
                vali_loss /= len(vali_loader)

            mae, rmse, mape, mpjpe = compute_metrics(y.cpu().detach().numpy(), output.cpu().detach().numpy())
            print('\nValidation Loss:', vali_loss, 'MAE:', mae, 'RMSE:', rmse, 'MAPE:', mape, 'MPJPE:', mpjpe, '\n')
            validation_log.append([vali_loss, mae, rmse, mape, mpjpe])
    

    # test the model 
    model.eval()
    test_loss = 0
    predict_list = []
    y_list = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            if configs.model_name == 'ST_Transformer':
                output = model(x, adj_matrix)
            else:
                output = model(x)
            loss = criterion(output, y)
            test_loss += loss.item()
            predict_list.append(output.cpu().detach().numpy())
            y_list.append(y.cpu().detach().numpy())
        test_loss /= len(test_loader)
        print('Test Loss:', test_loss)
    
    predict_data = np.concatenate(predict_list, axis=0)
    y_data = np.concatenate(y_list, axis=0)

    save_dir = f'test_results/{configs.model_name}/'
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    
    np.save(save_dir + 'predict_data.npy', predict_data)
    np.save(save_dir + 'y_data.npy', y_data)
    np.save(save_dir + 'scalar.npy', scaler)

    # save the log
    training_log = np.array(training_log)
    validation_log = np.array(validation_log)

    if os.path.exists(f'log/{configs.model_name}') == False:
        os.makedirs(f'log/{configs.model_name}')

    np.save(f'log/{configs.model_name}/training_log.npy', training_log)
    np.save(f'log/{configs.model_name}/validation_log.npy', validation_log)

    

if __name__ == "__main__":
    main()