import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .ii_transformer import TransformerModule
from .iii_callback import ROI



def train(args, train_dataloader, validation_dataloader, test_dataloader):
    # define module 
    transformer = TransformerModule(args) # make it as argment.

    # train - grid search?
    result_root = "/home/kyuholee/SPD_ver_2/D_result"

    # create callback for ROI 
    # ho does pytorch go through the test dataset? by order? or if it is not in order, how can i compute ROI? 
    ROI_callback = ROI()

    # callback for ckpt
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="validation_loss",
        mode="min",
        dirpath=(result_root + "/checkpoints"),
        filename="Model-{epoch:02d}",
    )

    #logger
    wandb_logger = WandbLogger(project="Stock Prediction", 
            save_dir=result_root)

    # train
    trainer = pl.Trainer(
        devices=4, 
        accelerator="gpu", 
        max_epochs=10,
        log_every_n_steps=10,
        logger=wandb_logger, 
        default_root_dir=result_root,
        callbacks=[checkpoint_callback, ROI_callback])

    # auto lr finder
    #tuner = Tuner(trainer)
    #tuner.lr_find(model=transformer, 
    #    train_dataloaders=train_dataloader, 
    #    val_dataloaders=validation_dataloader,
    #    max_lr=0.0001,
    #    min_lr=1.0e-8)

    trainer.fit(model=transformer, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    # test
    trainer.test(model=transformer, dataloaders=test_dataloader)
    
    wandb.finish()