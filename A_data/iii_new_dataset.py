import torch
from torch.utils.data import Dataset
from .i_download import download

import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from pathlib import Path

root = '/home/kyuholee/SPD_ver_2/C_cache'

label_data = Path(root) / "label"
label_data.mkdir(parents=True, exist_ok=True)

def r_value_calc(df):

    open = df['Open'].iloc[0]
    close = df['Close'].iloc[-1]
    
    return (close - open)/open

class newStockDataset(Dataset):

    def __init__(self, arg):
        
        # baisc argument
        self.arg = arg
        
        # all datas
        self.list_of_coin_datas = []
        
        # length related
        self.cumulative_length = []
        self.total_length = 0

        # download the raw data and compute total length
        for sym in arg['coin list']:
            
            # download
            data, _ = download(sym, arg['timeframe'], arg['start'], arg['end'])
            
            # process n/a values
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data = data.dropna(ignore_index=True)
            
            # store data
            self.list_of_coin_datas.append(data)
            
            # compute length of each data
            length = len(data) - (self.arg['x_frame'] + self.arg['y_frame']) + 1
            if length < 1:
                raise NameError("x and y frames are too big")
            
            # store stats 
            self.total_length = self.total_length + length
            self.cumulative_length.append(self.total_length)
        
        # if return value exist, just use that
        if os.path.exists(label_data/ (f'{arg["start"]}_{arg["end"]}_label.pkl')):
            print('data already exist')
            self.r_value = pd.read_pickle(label_data/ (f'{arg["start"]}_{arg["end"]}_label.pkl'))
        else:
            # load entire return value
            self.r_value = pd.Series([self.__perWindow__(i)[2] for i in tqdm(range(self.total_length))])
            self.r_value.to_pickle(label_data/ (f'{arg["start"]}_{arg["end"]}_label.pkl'))
        
        # compute alpha and beta
        intl_hold = 0.85  # marks the threshold of hold strip
        intl_buy_sell = 0.997  # marks the buy/sell upper/lower limits
        
        self.alpha = self.r_value.abs().quantile(intl_hold)
        self.beta = self.r_value.abs().quantile(intl_buy_sell)
        
        # compute label 1 
        self.label_1 = pd.Series(1, index=self.r_value.index)
        self.label_1[self.r_value > self.arg['revenue']] = 0 
        self.label_1[self.r_value < (-1 * self.arg['revenue'])] = 2
        
        # compute label 2
        self.label_2 = pd.Series(1, index=self.r_value.index)
        self.label_2[(self.r_value < self.beta) & (self.r_value > self.alpha)] = 0
        self.label_2[(self.r_value > -self.beta) & (self.r_value < -self.alpha)] = 2
        

    def __len__(self):    
        return self.total_length
    
    def __perWindow__(self, idx):
        # which coin does it belongs to? 
        coin_index = 0
        data_idx = idx 
        for i, l in enumerate(self.cumulative_length):
            if idx >= l:
                coin_index = i + 1
                data_idx = idx - l
        
        # reloacate data index 
        data_idx = data_idx + self.arg['x_frame']
        
        # load that coin data
        data = self.list_of_coin_datas[coin_index]
        
        # create window
        window = data.iloc[data_idx-self.arg['x_frame']:data_idx+self.arg['y_frame']].copy(deep=True) 
        
        # divide X and y 
        X = window[:self.arg['x_frame']]
        y = window[self.arg['x_frame']:]
        
        # compute return value on y
        ret = r_value_calc(y)
        
        # make y numpy
        y = y.drop(columns=['Time'])
        y = y.values
        
        # normalize X
        ohlcv = X[['High', 'Low', 'Open', 'Close', 'Volume']]
        ohlcv = ohlcv.apply(lambda x: np.log((x+1)) - np.log(x.iloc[self.arg['x_frame']-1]+1))
        #ohlcv = ohlcv.apply(lambda x: (np.log(x+1) - np.log(x+1).mean())/ (np.log(x+1).std()))
        X.loc[:,['High', 'Low', 'Open', 'Close', 'Volume']] = ohlcv 
        
        # isolate time from X
        X = X.drop(columns=['Time'])
        X = X.values
        
        return X, y, ret

    def __getitem__(self, idx):
        X, y, r = self.__perWindow__(idx)
        
        if self.arg['label'] == 1:
            label = self.label_1[idx]
        elif self.arg['label'] == 2:
            label = self.label_2[idx]
        
        return torch.FloatTensor(X), torch.FloatTensor(y), torch.tensor(label, dtype=torch.int64)
    
    def get_labels(self):
        if self.arg['label'] == 1:
            return self.label_1
        elif self.arg['label'] == 2:
            return self.label_2

# create dataset for original paper -> just return a line of data, processed to have
# label, and no OHLCV, Time. 

if __name__ == "__main__" : 
    
    data_args = {
    'label' : 2,     
    'coin list': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'], 
    'timeframe': '5m', 
    'start': (2020, 1, 1, 10), 
    'end': (2020, 2, 1, 10),
    
    'x_frame': 100, 
    'y_frame': 5, 
    'revenue': 0.015, 
    
    'batch size': 100
    }
    
    A = newStockDataset(data_args)
    