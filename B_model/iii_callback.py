from typing import Any
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

import pandas as pd

class ROI(Callback):
    def __init__(self):
        self.state = {"ROI": 1, "Position_open": 0}

    @property
    def state_key(self) -> str:
        # note: we do not include `verbose` here on purpose
        return "ROI"

    def on_test_batch_end(self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs: STEP_OUTPUT, 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0):
        
        # load y data
        X, y, target = batch
        y = y.squeeze(0).cpu().numpy()
        column_names = [
          'Open',         
          'High',         
          'Low',         
          'Close',         
          'Volume',         
          'RSI',         
          'boll',         
          'ULTOSC',         
          'zsVol',         
          'PR_MA_Ratio_short',         
          'MA_Ratio_short',         
          'MA_Ratio',         
          'PR_MA_Ratio',         
          'DayOfWeek',         
          'Month',         
          'Hourly'     
          ]
        y = pd.DataFrame(y, columns=column_names)
        
        # load prediction
        _ , pred = outputs.max(dim=1)
        
        # what label to use?
        label = pl_module.args['label']
        # what is stop loss?
        stop_loss = pl_module.args['stop_loss']
        # what is fee? 
        
        # what is revenue?
        revenue = pl_module.args['revenue']
        
        if label == 1 :
            if pred == 0:
                # price increased, check y to compute ROI 
                open_price = y['Open'].iloc[0]
                close_price = y['Close'].iloc[-1]
                
                roi = close_price / open_price
                
                # update ROI, if no investment is currently on.
                if self.state["Position_open"] > 0:
                    self.state["Position_open"] = self.state["Position_open"] - 1
                else:
                    self.state["ROI"] = self.state["ROI"] * roi
                    self.state["Position_open"] = 5
                
            else:
                # price stay, or decrease, so no investment made
                pass
            
            
        elif label == 2:
            
            pass
        
        pl_module.log("ROI", self.state["ROI"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()