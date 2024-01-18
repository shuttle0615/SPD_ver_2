MAX_LEN = 100
import wandb

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.tuner import Tuner

import torchmetrics
from torchmetrics import Metric

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

from .i_transformer import TransformerModule 
from .train import train