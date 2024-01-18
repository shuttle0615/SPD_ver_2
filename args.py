# list the args to test 



template = {
    'label' : 2,
    
    'coin list': ['BTC/USDT'], 
    'timeframe': '5m', 
    'start': (2020, 1, 1, 10), 
    'end': (2023, 1, 1, 10),
    
    'x_frame': 100, 
    'y_frame': 5, 
    'revenue': 0.015, 
    
    'data ratio': [0.7, 0.9],
    
    'batch size': 1000,
    
    "nhid_tran" : 256, #model
    "nhead" : 16, #model
    "nlayers_transformer" : 8, #model
    "attn_pdrop" : 0.1, #model
    "resid_pdrop" : 0.1, #model
    "embd_pdrop" : 0.1, #model
    "nff" : 4 * 256, #model
    
    "epoch": 10,
    "lr": 0.0005
}

lr = []