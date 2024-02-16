import talib
import pandas as pd

def technical_indicator(data):
    '''
    create features for technical indicator
    '''
    
    # oscilaration indicators
    data['RSI'] = ((talib.RSI(data['Close'])) / 100)
    upper_band, _, lower_band = talib.BBANDS(data['Close'], nbdevup=2, nbdevdn=2, matype=0)
    data['boll'] = ((data['Close'] - lower_band) / (upper_band - lower_band))
    data['ULTOSC'] = ((talib.ULTOSC(data['High'], data['Low'], data['Close'])) / 100)
    #data['pct_change'] = (data['Close'].pct_change())
    data['zsVol'] = (data['Volume'] - data['Volume'].mean()) / data['Volume'].std()
    data['PR_MA_Ratio_short'] = \
        ((data['Close'] - talib.SMA(data['Close'], 21)) / talib.SMA(data['Close'], 21))
    data['MA_Ratio_short'] = \
        ((talib.SMA(data['Close'], 21) - talib.SMA(data['Close'], 50)) / talib.SMA(data['Close'], 50))
    data['MA_Ratio'] = (
                (talib.SMA(data['Close'], 50) - talib.SMA(data['Close'], 100)) / talib.SMA(data['Close'], 100))
    data['PR_MA_Ratio'] = ((data['Close'] - talib.SMA(data['Close'], 50)) / talib.SMA(data['Close'], 50))
    
    # date related
    data['DayOfWeek'] = pd.to_datetime(data['Time']).dt.dayofweek
    data['Month'] = pd.to_datetime(data['Time']).dt.month
    data['Hourly'] = pd.to_datetime(data['Time']).dt.hour
    
    # find pattern
    '''data['CDL2CROWS'] = talib.CDL2CROWS(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDL3INSIDE'] = talib.CDL3INSIDE(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLBELTHOLD'] = talib.CDLBELTHOLD(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLDOJISTAR'] = talib.CDLDOJISTAR(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLENGULFING'] = talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLHIKKAKE'] = talib.CDLHIKKAKE(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLKICKING'] = talib.CDLKICKING(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLLONGLINE'] = talib.CDLLONGLINE(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLMARUBOZU'] = talib.CDLMARUBOZU(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLMATHOLD'] = talib.CDLMATHOLD(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLONNECK'] = talib.CDLONNECK(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLPIERCING'] = talib.CDLPIERCING(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLSHORTLINE'] = talib.CDLSHORTLINE(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLTHRUSTING'] = talib.CDLTHRUSTING(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLTRISTAR'] = talib.CDLTRISTAR(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(data['Open'], data['High'], data['Low'], data['Close']) / 100
    data['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(data['Open'], data['High'], data['Low'], data['Close']) / 100
    # data['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(data['Open'], data['High'], data['Low'], data['Close']) / 100'''

    return data


if __name__ == "__main__" : 
    from .i_download import download
    # get data from cache
    df, name = download("BTC/USDT", '1h', (2022, 1, 10, 1), (2022, 1, 12, 2))
    # alread integrated
    
    # print
    print(df.head())
    print(df.info())
    
    