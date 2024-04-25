import torch
import random
import numpy as np
from param import configs

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_dataset():
    data = np.load('dataset/C001/A001.npy')

    # normalize the data
    data = (data - np.mean(data, axis=(0,1))) / np.std(data, axis=(0,1))

    # shuffle the data 
    original_index = np.arange(data.shape[0])
    np.random.shuffle(original_index)
    data = data[original_index]

    # split the data
    train_x, train_y = data[:int(len(data)*configs.train_radio), :60, :], data[:int(len(data)*configs.train_radio), 60:, :]
    vali_x, vali_y = data[int(len(data)*configs.train_radio):int(len(data)*(configs.train_radio+configs.vali_radio)), :60, :], data[int(len(data)*configs.train_radio):int(len(data)*(configs.train_radio+configs.vali_radio)), 60:, :]
    test_x, test_y = data[int(len(data)*(configs.train_radio+configs.vali_radio)):, :60, :], data[int(len(data)*(configs.train_radio+configs.vali_radio)):, 60:, :]

    print('train_x shape:', train_x.shape, 'train_y shape:', train_y.shape)
    print('vali_x shape:', vali_x.shape, 'vali_y shape:', vali_y.shape)
    print('test_x shape:', test_x.shape, 'test_y shape:', test_y.shape)

    # convert to tensor
    train_x, train_y = torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32)
    vali_x, vali_y = torch.tensor(vali_x, dtype=torch.float32), torch.tensor(vali_y, dtype=torch.float32)
    test_x, test_y = torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    vali_dataset = torch.utils.data.TensorDataset(vali_x, vali_y)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)
    vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=configs.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False)

    return train_loader, vali_loader, test_loader

