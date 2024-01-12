from A_data import *

class DataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        # initialize the data relaged arguments
        kwargs
        
        pass

    def prepare_data(self):
        # raw data to 
        
        pass        

    def setup(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass
    def test_dataloader(self):
        pass
    def predict_dataloader(self):
        pass


if __name__ == "__main__" : 
    
    testing_arg = {
        "ticker" : ['BTC/USDT'],
        "timeframe" : '5m',
        
        "start date" : (2020,10,1,10),
        "end date" : (2023,10,1,10),
        "x_frames" : 50,
        "y_frames" : 5,
        
        "validation ratio" : 0.7,
        "test ratio" : 0.9,
        "batch size" : 200  
    }
    
    DM = DataModule(testing_arg)