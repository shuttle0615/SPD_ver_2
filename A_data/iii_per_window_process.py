from A_data import *

# before we start, we need to drop N/A values
# data need to be seperated for train, test, validation. 
# check how they did evaluation part


def r_value_calc(df, x_frames):
    
    df_y = df.iloc[x_frames:]

    open = df_y['Open'].iloc[0]
    close = df_y['Close'].iloc[-1]
    
    return (close - open)/open


def per_window_process(df, name, x_frames, y_frames, rev):
    # name
    new_name = name + f'X:{x_frames}_y:{y_frames}_r:{rev}_processed'
    
    # read cache, if exist.
    if os.path.exists(processed_data / (new_name + '.pkl')):
        print(f'{new_name} : data already processed')
        return pd.read_pickle(processed_data / (new_name + '.pkl')), new_name
    
    # drop all N/A values from df
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    
    # initialize data 
    X = []
    y = []
    r_value = []
    time = []

    # length of data
    l = len(df) - (x_frames + y_frames) + 1
    if l < 1:
        raise NameError("x and y frames are too big")
    
    print("data processing start")
    # process data
    for idx in tqdm(range(x_frames, x_frames + l)):
        data = df.iloc[idx-x_frames:idx+y_frames].copy(deep=True)
        
        # compute how much return was gained 
        data1 = data[['High', 'Low', 'Open', 'Close', 'Volume']]
        r_value.append(r_value_calc(data1, x_frames))
        
        # data normalization
        data1 = data1.apply(lambda x: np.log((x+1)) - np.log(x.iloc[x_frames-1]+1))
        #data1 = data1.apply(lambda x: (np.log(x+1) - np.log(x+1).mean())/ (np.log(x+1).std()))
        data.loc[:,['High', 'Low', 'Open', 'Close', 'Volume']] = data1 
        
        # save time seperately
        time.append(data['Time'].values.tolist())
        
        # save data
        data = data.drop(columns=['Time'])
        data = data.values
        X.append(data[:x_frames].tolist())
        y.append(data[x_frames:].tolist())
    
    # compute alpha and beta
    intl_hold = 0.85  # marks the threshold of hold strip
    intl_buy_sell = 0.997  # marks the buy/sell upper/lower limits
    
    r_value = pd.Series(r_value)
    alpha = r_value.abs().quantile(intl_hold)
    beta = r_value.abs().quantile(intl_buy_sell)
    
    # append label_1
    label_1 = pd.Series(1, index=r_value.index)
    label_1[r_value > rev] = 0 
    label_1[r_value < (-1 * rev)] = 2
    
    # append label_2 
    label_2 = pd.Series(1, index=r_value.index)
    label_2[(r_value < beta) & (r_value > alpha)] = 0
    label_2[(r_value > -beta) & (r_value < -alpha)] = 2
    
    # save data
    new_df = pd.DataFrame({'time': time, 'X': X, 'y': y, 'label_1': label_1, 'label_2': label_2})
    new_df.to_pickle(processed_data / (new_name + '.pkl'))
    
    return new_df, new_name


if __name__ == "__main__" : 
    from i_download import download
    df, name = download("BTC/USDT", '1h', (2022, 1, 10, 1), (2022, 7, 12, 2))    
    
    # first load    
    new_df, new_name = per_window_process(df, name, 8, 2, 0.01)
    
    print(new_df.head())
    new_df.info()
    print(len(new_df))
    
    # second load - pickle does preserve data
    new_df, new_name = per_window_process(df, name, 8, 2, 0.01)
    print(type((new_df['time'].iloc[0][0])))
    
    # visulaize the data 