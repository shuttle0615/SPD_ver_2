from __init__ import *

def technical_indicator(data):
    
    data['RSI'] = ((talib.RSI(data['Close'])) / 100)
    upper_band, _, lower_band = talib.BBANDS(data['Close'], nbdevup=2, nbdevdn=2, matype=0)
    data['boll'] = ((data['Close'] - lower_band) / (upper_band - lower_band))
    data['ULTOSC'] = ((talib.ULTOSC(data['High'], data['Low'], data['Close'])) / 100)
    data['zsVol'] = (data['Volume'] - data['Volume'].mean()) / data['Volume'].std()
    data['PR_MA_Ratio_short'] = \
        ((data['Close'] - talib.SMA(data['Close'], 21)) / talib.SMA(data['Close'], 21))
    data['MA_Ratio_short'] = \
        ((talib.SMA(data['Close'], 21) - talib.SMA(data['Close'], 50)) / talib.SMA(data['Close'], 50))
    data['MA_Ratio'] = (
                (talib.SMA(data['Close'], 50) - talib.SMA(data['Close'], 100)) / talib.SMA(data['Close'], 100))
    data['PR_MA_Ratio'] = ((data['Close'] - talib.SMA(data['Close'], 50)) / talib.SMA(data['Close'], 50))

    return data


