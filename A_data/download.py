from __init__ import *

def download(symbol, timeframe, start, end):
    '''
    download data and save it at D_data folder
    '''
    name = f'{symbol.replace("/", ":")}_{timeframe}_{start}_{end}'
    
    # if cache exist, load and finsh.
    if os.path.exists(cache_dir / (name + '.pkl')):
        return pd.read_pickle(cache_dir / (name + '.pkl')), name
    
    # create binance object
    binance = ccxt.binance()  
    
    # process starting time
    startTime = datetime(*start)
    startTime = datetime.timestamp(startTime)
    startTime = int(startTime*1000) 
    
    # process ending time
    endTime = datetime(*end)
    endTime = datetime.timestamp(endTime)
    endTime = int(endTime*1000) 
    
    # process time frame
    if timeframe == '1h':
        unit_time = (60*60*1000)
    elif timeframe == '30m':
        unit_time = (30*60*1000)
    elif timeframe == '15m':
        unit_time = (15*60*1000)
    elif timeframe == '5m':
        unit_time = (5*60*1000)
    elif timeframe == '1m':
        unit_time = (60*1000)
    else:
        raise NameError("not supported time frame")
    
    # compute the length btw start and end time (how much data points requried?)
    diff = endTime - startTime
    num_unit_time = diff // unit_time

    # only 1000 data points are available per requrest
    if num_unit_time > 1000 :
        # over 1000 data points

        repeat = num_unit_time // 1000 
        leftover = num_unit_time % 1000 

        ohlcv = []

        for i in range(repeat):
            ohlcv = ohlcv + binance.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=(startTime + i*unit_time*1000), limit=1000)

        ohlcv = ohlcv + binance.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=(startTime + repeat*unit_time*1000), limit=leftover)    

    else: 
        # under 1000 data points
        ohlcv = binance.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=startTime, limit=num_unit_time)

    # process the data
    df = pd.DataFrame(ohlcv, columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Time'] = [datetime.fromtimestamp(float(time)/1000) for time in df['Time']]
    df.set_index('Time', inplace=True)
    
    #save the cache
    df.to_pickle(cache_dir / (name + '.pkl'))

    return df, name

if __name__ == "__main__" : 
    
    #first attempt
    df, name = download("BTC/USDT", '1h', (2022, 1, 10, 1), (2022, 1, 12, 2))
    
    #second attempt
    df, name = download("BTC/USDT", '1h', (2022, 1, 10, 1), (2022, 1, 12, 2))
    
    #display
    print(df.head())
    print(name)