import pandas as pd

from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from A_data import data_args, data_exp_args
from A_data import StockDataset
from A_data import concatanation

from B_model import model_args
from B_model import TransformerModule

# download all data
train, validation, test = concatanation(data_args)
#train, validation, test = concatanation(data_exp_args)
# create dataset
train_dataset = StockDataset(train)
validation_dataset = StockDataset(validation)
test_dataset = StockDataset(test)

# create dataloader 
train_dataloader = DataLoader(train_dataset, batch_size=data_exp_args['batch size'], sampler=ImbalancedDatasetSampler(train_dataset)) 
validation_dataloader = DataLoader(validation_dataset, batch_size=data_exp_args['batch size'], sampler=ImbalancedDatasetSampler(validation_dataset)) 
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# define module 
transformer = TransformerModule(model_args)

# train - grid search?
result_root = "/home/kyuholee/SPD_ver_2/D_result"

wandb_logger = WandbLogger(project="Stock Prediction", log_model="all", save_dir=result_root)
trainer = pl.Trainer(devices="auto", 
    accelerator="auto", 
    max_epochs=10,
    log_every_n_steps=10,
    logger=wandb_logger, 
    default_root_dir=result_root)

trainer.fit(model=transformer, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)




# 

