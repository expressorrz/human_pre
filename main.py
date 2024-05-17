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

    print('Model Name:', configs.model_name)

    # model selection
    if configs.model_name == 'MLP':
        model = MLP_model(configs.input_dim, configs.hidden_dim1, configs.hidden_dim2, configs.output_dim).to(device)
    elif configs.model_name == 'CNN':
        model = CNN_model(configs.input_dim, configs.hidden_dim2, configs.hidden_dim_fc, configs.output_dim).to(device)
    elif configs.model_name == 'GRU':
        model = GRU_model(configs.input_dim, configs.hidden_dim2, configs.hidden_dim_fc, configs.num_layers, configs.output_dim).to(device)
    elif configs.model_name == 'LSTM':
        model = LSTM_model(configs.input_dim, configs.hidden_dim1, configs.hidden_dim2, configs.hidden_dim_fc, configs.num_layers, configs.output_dim).to(device)
    elif configs.model_name == 'Transformer':
        model = Transformer(input_size=configs.input_dim, hidden_dim=configs.hidden_dim_trans, num_layers=configs.num_layers, output_size=configs.output_dim).to(device)

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
                    output = model(x)
                    loss = criterion(output, y)
                    vali_loss += loss.item()
                vali_loss /= len(vali_loader)


            print('Epoch:', epoch, 'Train Loss:', train_loss, 'Validation Loss:', vali_loss)
            validation_log.append(vali_loss)
    

    # test the model 
    model.eval()
    test_loss = 0
    predict_list = []
    y_list = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
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