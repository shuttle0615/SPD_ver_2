from A_data import *

# the problem is that each ticker takes too long... just choose five top volume
'''
# get all tickers end with usdt
binance = ccxt.binance() 
ls = binance.fetch_tickers()
asset = [coin for coin in ls if coin.endswith("USDT")]

# sort by size of volume
volumels = [{"symbol":ls[coin]['symbol'], "volume":ls[coin]['quoteVolume']} for coin in asset]
df = pd.DataFrame(volumels).sort_values(by=['volume'], ascending=False)

# get all coin tickers except for stable coins, and has volume larger than 1.0e+0.7
stable_coin_ls = ["USDC/USDT", "DAI/USDT", "TUSD/USDT", "FDUSD/USDT", "USDD/USDT", 
 "BUSD/USDT", "USDP/USDT", "PYUSD/USDT", "USTC/USDT", "EUR/USDT"]

df_good = df[(~df["symbol"].isin(stable_coin_ls)) & (df["volume"] > 1.0e+07) ]
list_of_coin = df_good["symbol"].values.tolist()
'''

def concatanation(arg):
    # if train, validation, test already exist, load them.
    if os.path.exists(Path(root) / 'train_dataset.pkl') and \
        os.path.exists(Path(root) / 'validation_dataset.pkl') and \
        os.path.exists(Path(root) / 'test_dataset.pkl'):
        print('datas already exist')
        
        train_dataset = pd.read_pickle(Path(root) / 'train_dataset.pkl')
        validation_dataset = pd.read_pickle(Path(root) / 'validation_dataset.pkl')
        test_dataset = pd.read_pickle(Path(root) / 'test_dataset.pkl')
        
        return train_dataset, validation_dataset, test_dataset
    
    # download all data for each symbol
    ls_train = []
    ls_validation = []
    ls_test = []
    
    for sym in arg['coin list']:
        print(f'downloading {sym} data')
        # download data
        df, name = download(sym, arg['timeframe'], arg['start'], arg['end'])
        # process data
        pr_df, _ = per_window_process(df, name, arg['x_frame'], arg['y_frame'], arg['revenue'])
        
        # devide data into train, validaition, test
        last = len(pr_df)
        idx1 = int(last * arg["data ratio"][0])
        idx2 = int(last * arg["data ratio"][1])
        
        print("train appending")
        ls_train.append(pr_df.iloc[0:idx1])
        print("validation appending")
        ls_validation.append(pr_df.iloc[idx1:idx2])
        print("test appending")
        ls_test.append(pr_df.iloc[idx2:last])
    
    train_dataset = pd.concat(ls_train, ignore_index=True)
    train_dataset.to_pickle(Path(root) / 'train_dataset.pkl')
    
    validation_dataset = pd.concat(ls_validation, ignore_index=True)
    validation_dataset.to_pickle(Path(root) / 'validation_dataset.pkl')
    
    test_dataset = pd.concat(ls_test, ignore_index=True)
    test_dataset.to_pickle(Path(root) / 'test_dataset.pkl')

    return train_dataset, validation_dataset, test_dataset

    
if __name__ == "__main__" : 
    
    exp_arg = {
        'coin_list': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XAI/USDT'],
        'timeframe': '5m', 
        'start': (2020, 1, 1, 10), 
        'end': (2020, 1, 2, 10),
        
        'x_frame': 8, 
        'y_frame': 2, 
        'revenue': 0.015, 
        'data ratio': [0.7, 0.9]
    }
    
    tr, va, te = concatanation(exp_arg)
    
    print("head of tr, va, te")
    print(tr.head())
    print(va.head())
    print(te.head())
    
    print("info of tr, va, te")
    tr.info()
    va.info()
    te.info()
    
    print("length of tr, va, te")
    print(len(tr))
    print(len(va))
    print(len(te))
        
        
    
    


