from A_data.iii_new_dataset import newStockDataset
from B_model.i_train import train

from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler


train_data_args = {
    'label' : 2,     
    'coin list': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'], 
    'timeframe': '5m', 
    
    'start': (2020, 9, 1, 10), 
    'end': (2022, 9, 1, 10),
    
    'x_frame': 100, 
    'y_frame': 5, 
    'revenue': 0.015, 
    
    'batch size': 100
}

validation_data_args = {
    'label' : 2,     
    'coin list': ['BTC/USDT', 'ETH/USDT'], 
    'timeframe': '5m', 
    
    'start': (2019, 4, 1, 10), 
    'end': (2019, 10, 1, 10),
    
    'x_frame': 100, 
    'y_frame': 5, 
    'revenue': 0.015, 
    
    'batch size': 100
}

test_data_args = {
    'label' : 2,     
    'coin list': ['BTC/USDT'], 
    'timeframe': '5m', 
    
    'start': (2023, 3, 1, 10), 
    'end': (2023, 9, 1, 10),
    
    'x_frame': 100, 
    'y_frame': 5, 
    'revenue': 0.015, 
    
    'batch size': 1
}

tr = newStockDataset(train_data_args)
va = newStockDataset(validation_data_args)
te = newStockDataset(test_data_args)

train_dataloader = DataLoader(tr, batch_size=train_data_args['batch size'], sampler=ImbalancedDatasetSampler(tr), num_workers=4) 
validation_dataloader = DataLoader(va, batch_size=validation_data_args['batch size'], sampler=ImbalancedDatasetSampler(va), num_workers=4) 
test_dataloader = DataLoader(te, batch_size=1, shuffle=False)

'''
# 0.0005, 0.0004, 0.0003, 0.0002, 0.0001
for lr in [0.0005, 0.0004, 0.0003, 0.0002, 0.0001]:
    args = {
        "nhid_tran" : 256, #model
        "nhead" : 16, #model
        "nlayers_transformer" : 8, #model
        "attn_pdrop" : 0.1, #model
        "resid_pdrop" : 0.1, #model
        "embd_pdrop" : 0.1, #model
        "nff" : 4 * 256, #model
        
        "epoch": 10,
        "lr": lr
    }

    train(args, tr, va, te)
'''