from A_data import *

class StockDataset(Dataset):

    def __init__(self, df, label_num=1):
        self.data = df
        
        self.all_X = self.data["X"].values.tolist()
        self.all_y = self.data[f"label_{label_num}"].values.tolist()
        self.all_time = self.data["time"].values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.FloatTensor(self.all_X[idx])
        y = torch.tensor(self.all_y[idx], dtype=torch.int64)
        return X, y
    
    def gettime(self, idx):
        return self.all_time[idx]
    
    def get_labels(self):
        return self.all_y

class simpleDataset(Dataset):
    pass