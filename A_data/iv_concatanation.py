from __init__ import *
from i_download import download
from iii_per_window_process import per_window_process

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
list_of_coin = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XAI/USDT']

def concatanation(list, arg):
    # download all data for each symbol
    for sym in list_of_coin:
        # download data
        df, name = download(sym, arg['timeframe'], arg['start'], arg['end'])
        # process data
        pr_df, name = per_window_process(df, name, arg['x_frame'], arg['y_frame'], arg['revenue'], arg['loss'])
        
        # devide data into train, validaition, test
        last = len(pr_df)
        idx1 = int(last * arg["data ratio"][0])
        idx2 = int(last * arg["data ratio"][1])
        
        train = pr_df.iloc[0:idx1]
        validation = pr_df.iloc[idx1:idx2]
        test = pr_df.iloc[idx2:last]
        
        
    # create train, validation, test list of dataframe, and concatanate each other
    # finally return df of train, test, validation set.
    
if __name__ == "__main__" : 
    exp_arg = {
        
    }
    
    pass
        
        
    
    


