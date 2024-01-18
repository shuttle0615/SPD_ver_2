from A_data import *

def dataloader(args):
    
    # download all data
    train, validation, test = concatanation(args)
    # create dataset
    train_dataset = StockDataset(train, label_num=args["label"])
    validation_dataset = StockDataset(validation, label_num=args["label"])
    test_dataset = StockDataset(test)

    # create dataloader 
    train_dataloader = DataLoader(train_dataset, batch_size=args['batch size'], sampler=ImbalancedDatasetSampler(train_dataset), num_workers=4) 
    validation_dataloader = DataLoader(validation_dataset, batch_size=args['batch size'], sampler=ImbalancedDatasetSampler(validation_dataset), num_workers=4) 
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_dataloader, validation_dataloader, test_dataloader