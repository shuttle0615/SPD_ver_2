import os
import ccxt
from tqdm import tqdm

from datetime import datetime
import numpy as np
import pandas as pd

from pathlib import Path

import talib as talib

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pytorch_lightning as pl

root = '/home/kyuholee/SPD_ver_2/C_cache'

raw_data = Path(root) / 'raw_data'
raw_data.mkdir(parents=True, exist_ok=True)

processed_data = Path(root) / 'processed_data'
processed_data.mkdir(parents=True, exist_ok=True)


data_args = {
    'coin list': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'], 
    'timeframe': '5m', 
    'start': (2020, 1, 1, 10), 
    'end': (2023, 1, 1, 10),
    
    'x_frame': 100, 
    'y_frame': 5, 
    'revenue': 0.015, 
    
    'data ratio': [0.7, 0.9],
    
    'batch size': 1000
}

data_exp_args = {
    'coin list': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'], 
    'timeframe': '5m', 
    'start': (2020, 1, 1, 10), 
    'end': (2020, 1, 2, 10),
    
    'x_frame': 100, 
    'y_frame': 5, 
    'revenue': 0.015, 
    
    'data ratio': [0.7, 0.9],
    
    'batch size': 10
}


from .ii_technical_indicator import technical_indicator
from .i_download import download
from .iii_per_window_process import per_window_process
from .iv_concatanation import concatanation
from .v_dataset import StockDataset
