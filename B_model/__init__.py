MAX_LEN = 100

import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model_args = {
    "nhid_tran" : 32, #model
    "nhead" : 8, #model
    "nlayers_transformer" : 3, #model
    "attn_pdrop" : 0.1, #model
    "resid_pdrop" : 0.1, #model
    "embd_pdrop" : 0.1, #model
    "nff" : 4 * 32, #model
    
    "epoch": 10,
    "lr": 0.0005
}

from .i_transformer import TransformerModule 