import os
import ccxt
from tqdm import tqdm
import time

from datetime import datetime
import numpy as np
import pandas as pd

from pathlib import Path

import talib as talib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchsampler import ImbalancedDatasetSampler

import pytorch_lightning as pl

root = '/home/kyuholee/SPD_ver_2/C_cache'

raw_data = Path(root) / 'raw_data'
raw_data.mkdir(parents=True, exist_ok=True)

processed_data = Path(root) / 'processed_data'
processed_data.mkdir(parents=True, exist_ok=True)


#'ETH/USDT', 'SOL/USDT', 'BNB/USDT' is gone..


from .ii_technical_indicator import technical_indicator
from .i_download import download
from .iii_per_window_process import per_window_process, simple_labeling
from .iv_concatanation import concatanation, simple_concatanation
from .v_dataset import StockDataset
from .vi_dataloader import dataloader
