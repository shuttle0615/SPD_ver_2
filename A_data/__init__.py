import os
import ccxt

from datetime import datetime
import numpy as np
import pandas as pd

from pathlib import Path

import talib as talib

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import pytorch_lightning as pl

root = '/home/kyuholee/SPD_ver_2/C_cache'

raw_data = Path(root) / 'raw_data'
raw_data.mkdir(parents=True, exist_ok=True)

processed_data = Path(root) / 'processed_data'
processed_data.mkdir(parents=True, exist_ok=True)
