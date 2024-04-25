import numpy as np
import torch
import time
import pandas as pd
from param import configs
from utils import *


str_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def main():
    # device
    device = torch.device(configs.device)

    # setup seed
    setup_seed(configs.seed)

    # load dateset
    df = pd.read_csv('raw_dataset/C001/A001/0.csv')
    print(df.to_numpy()[:,1:].shape)


if __name__ == "__main__":
    main()