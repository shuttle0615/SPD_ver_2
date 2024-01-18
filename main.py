from A_data import dataloader
from B_model import train

data_args = {
    'label' : 2,     
    'coin list': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'], 
    'timeframe': '5m', 
    'start': (2020, 1, 1, 10), 
    'end': (2020, 7, 1, 10),
    
    'x_frame': 100, 
    'y_frame': 5, 
    'revenue': 0.015, 
    
    'data ratio': [0.7, 0.9],
    
    'batch size': 100
}

tr, va, te = dataloader(data_args)

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
