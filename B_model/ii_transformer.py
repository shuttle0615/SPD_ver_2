from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from pytorch_lightning import LightningModule

import torchmetrics

MAX_LEN = 100
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class MaskedMultiheadAttention(nn.Module):
    """
    A vanilla multi-head masked attention layer with a projection at the end.
    """
    def __init__(self, args, mask=False):
        super(MaskedMultiheadAttention, self).__init__()
        assert args["nhid_tran"] % args["nhead"] == 0
        # mask : whether to use
        # key, query, value projections for all heads
        self.key = nn.Linear(args["nhid_tran"], args["nhid_tran"])
        self.query = nn.Linear(args["nhid_tran"], args["nhid_tran"])
        self.value = nn.Linear(args["nhid_tran"], args["nhid_tran"])
        # regularization
        self.attn_drop = nn.Dropout(args["attn_pdrop"])
        # output projection
        self.proj = nn.Linear(args["nhid_tran"], args["nhid_tran"])
        # causal mask to ensure that attention is only applied to the left in the input sequence
        if mask:
            self.register_buffer("mask", torch.tril(torch.ones(MAX_LEN, MAX_LEN)))
        self.nhead = args["nhead"]
        self.d_k = args["nhid_tran"] // args["nhead"]

    def forward(self, q, k, v, mask=None):
        # WRITE YOUR CODE HERE

        Q = self.query(q)
        Q_size = Q.size()
        Q = Q.reshape([Q_size[0], Q_size[1], self.nhead, -1])
        Q = torch.transpose(Q, 1, 2)

        K = self.key(k)
        K_size = K.size()
        K = K.reshape([K_size[0], K_size[1], self.nhead, -1])
        K = torch.transpose(K, 1, 2)

        V = self.value(v)
        V_size = V.size()
        V = V.reshape([V_size[0], V_size[1], self.nhead, -1])
        V = torch.transpose(V, 1, 2)

        K = torch.transpose(K, 2, 3)
        R = torch.matmul(Q, K)
        R = R/(self.d_k ** (1/2))

        #casual masking
        if hasattr(self, "mask"):
          temp_mask = self.mask[:R.size(2),:R.size(3)]
          temp_mask = temp_mask < 0.5
          R = R.masked_fill_(temp_mask, -float('Inf'))

        #Pad masking
        if mask != None:
          mask = mask < 0.5
          R = R.permute([2,1,0,3])
          R = R.masked_fill_(mask.to(device), -float('Inf'))
          R = R.permute([2,1,0,3])

        R = torch.nn.Softmax(dim=-1)(R)
        R = self.attn_drop(R)
        output = torch.matmul(R, V)

        output = output.transpose(1,2)
        output = output.reshape(output.size(0), output.size(1), -1)
        output = self.proj(output)

        assert output != None , output
        return output


class TransformerEncLayer(nn.Module):
    def __init__(self, args):
        super(TransformerEncLayer, self).__init__()
        self.ln1 = nn.LayerNorm(args["nhid_tran"])
        self.ln2 = nn.LayerNorm(args["nhid_tran"])
        self.attn = MaskedMultiheadAttention(args)
        self.dropout1 = nn.Dropout(args["resid_pdrop"])
        self.dropout2 = nn.Dropout(args["resid_pdrop"])
        self.ff = nn.Sequential(
            nn.Linear(args["nhid_tran"], args["nff"]),
            nn.ReLU(),
            nn.Linear(args["nff"], args["nhid_tran"])
        )

    def forward(self, x, mask=None):
        # WRITE YOUR CODE HERE
        output = self.ln1(x)
        output = output + self.dropout1(self.attn(output, output, output, mask))
        output = self.ln2(output)
        output = output + self.dropout2(self.ff(output))

        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, args, max_len=4096):
        super().__init__()
        dim = args["nhid_tran"]
        pos = np.arange(0, max_len)[:, None]
        i = np.arange(0, dim // 2)
        denom = 10000 ** (2 * i / dim)

        pe = np.zeros([max_len, dim])
        pe[:, 0::2] = np.sin(pos / denom)
        pe[:, 1::2] = np.cos(pos / denom)
        pe = torch.from_numpy(pe).float()

        self.register_buffer('pe', pe)

    def forward(self, x):
        # DO NOT MODIFY
        # 1 -> 0 but why?
        return x + self.pe[:x.shape[1]]

class TransformerEncoder(nn.Module):

    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        # input embedding stem
        self.tok_emb = nn.Linear(16, args["nhid_tran"]) #ohlev encoding
        self.pos_enc = PositionalEncoding(args)
        self.dropout = nn.Dropout(args["embd_pdrop"])
        # transformer
        self.nlayers_transformer = args["nlayers_transformer"]
        self.transform = nn.ModuleList([TransformerEncLayer(args) for _ in range(args["nlayers_transformer"])])
        # decoder head
        self.ln_f = nn.LayerNorm(args["nhid_tran"])
        self.classifier_head = nn.Sequential(
           nn.Linear(args["nhid_tran"], args["nhid_tran"]),
           nn.LeakyReLU(),
           nn.Dropout(args["embd_pdrop"]),
           nn.Linear(args["nhid_tran"], args["nhid_tran"]),
           nn.LeakyReLU(),
           nn.Linear(args["nhid_tran"], 3)
       )


    def forward(self, x, mask=None):
        # WRITE YOUR CODE HERE
        output = self.tok_emb(x)
        output = self.pos_enc(output)
        output = self.dropout(output)

        for i in range(self.nlayers_transformer):
          output = self.transform[i](output, mask=mask)

        output = self.ln_f(output)
        output = output.mean(dim=1)
        output = self.classifier_head(output)

        return output


class TransformerModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lr = self.args["lr"]
        self.transformer = TransformerEncoder(self.args)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=3)
        self.F1 = torchmetrics.F1Score(task="multiclass", num_classes=3)
        self.save_hyperparameters()
        
    def forward(self, inputs):
        return self.transformer(inputs)
    
    def training_step(self, batch, batch_idx):
        # can i get y values here? if so, it can compute ROI 
        inputs, _, target = batch
        output = self.transformer(inputs)
        loss = torch.nn.functional.cross_entropy(output, target)
        
        accuracy = self.accuracy(output, target)
        F1 = self.F1(output, target)
        
        self.log_dict({"loss":loss, "accuracy":accuracy, "F1":F1})
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, _, target = batch
        output = self.transformer(inputs)
        val_loss = torch.nn.functional.cross_entropy(output, target)
        
        val_accuracy = self.accuracy(output, target)
        val_F1 = self.F1(output, target)
        
        self.log_dict({"validation_loss":val_loss, "validation_accuracy":val_accuracy, "val_F1":val_F1}, sync_dist=True)
        return val_loss, val_accuracy
    
    def test_step(self, batch, batch_idx):
        inputs, y, target = batch
        output = self.transformer(inputs)
        test_loss = torch.nn.functional.cross_entropy(output, target)
        
        test_accuracy = self.accuracy(output, target)
        test_F1 = self.F1(output, target)
        
        self.log_dict({"test_loss":test_loss, "test_accuracy":test_accuracy, "test_F1":test_F1}, sync_dist=True)
        
        return output
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, 
            mode='min', 
            factor=0.5, 
            patience=1, 
            threshold=0.001, 
            threshold_mode='rel', 
            cooldown=0, 
            min_lr=0.00001,
            verbose=True)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'validation_loss',
                'strict': True,
            }
        }