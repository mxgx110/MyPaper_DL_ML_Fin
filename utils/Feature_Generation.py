import numpy as np
import pandas as pd
import talib
from arch import arch_model

class feature_engineering():

  def __init__(self, df, using_return = False, cols_if_true = ['close']):
    self.df = df
    
    if using_return:
      for i in range(len(cols_if_true)):
        returns = (np.log(self.df[cols_if_true[i]])).pct_change()
        self.df['Return_'+cols_if_true[i]] = returns

      (self.df).dropna(inplace = True)

  def SMA(self, window, col = 'close'):
    sma = talib.SMA(self.df[col], window)
    self.df["SMA_"+col+"_"+str(window)] = sma

  def SMA_ratio(self, window_1 = 5, window_2 = 15, col="close"):
    sma_1 = talib.SMA(self.df[col], window_1)
    sma_2 = talib.SMA(self.df[col], window_2)
    sma_ratio = sma_1/sma_2
    self.df["SMA_ratio_"+col+"_"+str(window_1)+"_"+str(window_2)] = sma_ratio

  def Parabolic_SAR(self,high_col = "high", low_col = "low", acceleration=0.02, maximum=0.2):
    psar = talib.SAR(self.df[high_col],self.df[low_col], acceleration, maximum)
    self.df["Parabolic_SAR"] = psar

  def ChaikinAD(self, high_col = "high", low_col = "low", close_col = "close", volume_col = "volume"):
    chaikinad = talib.AD(self.df[high_col],self.df[low_col],self.df[close_col],self.df[volume_col])
    self.df["ChaikinAD"] = chaikinad

  def MA_on_RSI(self, rsi_window_1 = 5, rsi_window_2 = 15, sma_window = 9,col = "close"):
    rsi_1 = talib.RSI(self.df[col], rsi_window_1)
    rsi_2 = talib.RSI(self.df[col], rsi_window_2)
    ratio = rsi_1/rsi_2
    ma_on_rsi = talib.SMA(ratio, sma_window)
    self.df["MA_on_RSI_"+str(rsi_window_1)+"_"+str(rsi_window_2)+"_"+str(sma_window)] = ma_on_rsi

  def ATR_ratio(self, window_1 = 5, window_2 = 15, high_col = 'high', low_col = 'low', col = "close"):
    atr_1 = talib.ATR(self.df[high_col], self.df[low_col], self.df[col], timeperiod=window_1)
    atr_2 = talib.ATR(self.df[high_col], self.df[low_col], self.df[col], timeperiod=window_2)
    atr_ratio = atr_1/atr_2
    self.df["ATR_ratio_"+col+"_"+str(window_1)+"_"+str(window_2)] = atr_ratio

  def RSI_ratio(self, window_1 = 5, window_2 = 15, col="close"):
    rsi_1 = talib.RSI(self.df[col], window_1)
    rsi_2 = talib.RSI(self.df[col], window_2)
    rsi_ratio = rsi_1/rsi_2
    self.df["RSI_ratio_"+col+"_"+str(window_1)+"_"+str(window_2)] = rsi_ratio

  
  def EMA(self, window, col = 'close'):
    ema = talib.EMA(self.df[col], timeperiod = window)
    self.df["EMA_"+col+"_"+str(window)] = ema

  def RSI(self, window, col = 'close'):
    rsi = talib.RSI(self.df[col], window)
    self.df["RSI_"+col+"_"+str(window)] = rsi 

  def MA(self, window, col = 'close'):
    ma = talib.MA(self.df[col], timeperiod= window)
    self.df["MA_"+col+"_"+str(window)] = ma 

  def MOM(self, window, col = 'close'):
    mom = talib.MOM(self.df[col], timeperiod= window)
    self.df["MTM_"+col+"_"+str(window)] = mom
  

  def ROC(self, window, col = 'close'):
    roc = talib.ROC(self.df[col], timeperiod= window)
    self.df["ROC_"+col+"_"+str(window)] = roc

  def MACD(self, col="close",fastperiod=12, slowperiod=26, signalperiod=9):
    macd , macdsignal, macdhist = talib.MACD(self.df[col], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod) 
    
    self.df["MACD"] = macd
    self.df["MACD_SIG"] = macdsignal
    self.df["MACD_HIST"] = macdhist


  def STOCHASTIC(self, high_col = 'high', low_col = 'low', col = "close", fastk_period = 5, fastd_period=3, fastd_matype=0, slowd_period=3, slowd_matype=0):
    
    FASTK, FASTD = talib.STOCHF(self.df[high_col], self.df[low_col], self.df[col], fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)
    SLOWK, SLOWD = talib.STOCH(self.df[high_col], self.df[low_col], self.df[col], fastk_period=fastk_period, slowk_period=fastd_period, slowk_matype=fastd_matype, slowd_period=slowd_period, slowd_matype=0)
    
    self.df["FASTK"] = FASTK
    self.df["FASTD"] = FASTD
    self.df["SLOWK"] = SLOWK
    self.df["SLOWD"] = SLOWD


  def BOLL(self, window = 20, col="close"):

    data = pd.DataFrame(None)
    data['15MA'] = self.df[col].transform(lambda x: x.rolling(window=window).mean())
    data['SD'] = self.df[col].transform(lambda x: x.rolling(window=window).std())
    self.df["BOLL_UPPER_"+str(window)] = data['15MA'] + 2*data['SD']
    self.df["BOLL_LOWER_"+str(window)] = data['15MA'] - 2*data['SD']


  def ATR(self,window, high_col = 'high', low_col = 'low', col = "close"):
    atr = talib.ATR(self.df[high_col], self.df[low_col], self.df[col], timeperiod=window)
    self.df["ATR_"+col+"_"+str(window)] = atr

  def MFI(self,window, high_col = 'high', low_col = 'low', col = "close", vol_col = "volume"):
    mfi = talib.MFI(self.df[high_col], self.df[low_col], self.df[col], self.df[vol_col], timeperiod=window)
    self.df["MFI_"+col+"_"+str(window)] = mfi 

  def WILLR(self,window, high_col = 'high', low_col = 'low', col = "close"):
    willr = talib.WILLR(self.df[high_col], self.df[low_col], self.df[col], timeperiod=window)
    self.df["WILLR_"+col+"_"+str(window)] = willr

  def ADX(self,window, high_col = 'high', low_col = 'low', col = "close"):
    adx = talib.ADX(self.df[high_col], self.df[low_col], self.df[col], timeperiod=window)
    self.df["ADX_"+col+"_"+str(window)] = adx
  
  def CCI(self,window, high_col = 'high', low_col = 'low', col = "close"):
    cci = talib.CCI(self.df[high_col], self.df[low_col], self.df[col], timeperiod=window)
    self.df["CCI_"+col+"_"+str(window)] = cci
  
  def Wilder(self, window, col = "close"):
    start = np.where(~np.isnan(self.df[col]))[0][0] 
    Wilder = np.array([np.nan]*len(self.df[col]))
    Wilder[start+window-1] = self.df[col][start:(start+window)].mean() 
    for i in range(start+window,len(self.df[col])):
        Wilder[i] = (Wilder[i-1]*(window-1) + self.df[col][i])/window 
    self.df["Wilder_"+col+"_"+str(window)] = Wilder
    
  def VWMA(self, window, col="close", weighted_col="volume", plot = False):
    vwma = [] 
    length = (self.df).shape[0] - window + 1
    for i in range(0,length):
      data = (self.df).iloc[i : i + window ,:]
      avg = np.average(data[col], weights=data[weighted_col])
      vwma.append(avg)

    for i in range(window - 1):
      vwma.insert(i, None)

    if plot:
      f = go.FigureWidget()
      f.add_scatter(x = list(range(len(self.df))), y = self.df[col], name = "Actual Data")
      f.add_scatter(x = list(range(len(self.df))), y = vwma, name = f"VWMA (window:{window})")
      f.show()
    
    self.df["VWMA_"+col+"_"+str(window)] = vwma

  def WVAD(self, window, open_col = "open", high_col = "high", low_col = "low", close_col = "close", volume_col = "volume"):
    wvad = [] 
    length = (self.df).shape[0] - window + 1
    for i in range(0,length):
      data = (self.df).iloc[i : i + window ,:]
      avg = np.average((data[open_col] - data[close_col])/(data[high_col] - data[low_col]), weights=data[volume_col])
      wvad.append(avg)

    for i in range(window - 1):
      wvad.insert(i, None)

    self.df["WVAD_"+str(window)] = wvad


  def GARCH(self, col = "close"):

    returns = (self.df[col]).pct_change().dropna()

    model = arch_model(returns, vol = 'GARCH', p = 1, q = 1)
    model_fit = model.fit(disp = "off")
    yhats = np.sqrt(model_fit.forecast(start = 0).variance)

    self.df["GARCH_"+col] = yhats


  def relative(self, window = 21, col = "volume"):


    sma = talib.SMA(self.df[col], window)
    self.df["Relative"+col+"_"+str(window)] = (self.df)[col][window-1:]/sma


  def must_be_used(self):

    self.df.dropna(inplace = True)

    for i in range(len((self.df).columns)):
      for j in range(len(self.df)):

        (self.df).iloc[j,i] = (self.df).iloc[j,i].astype(np.float64)

      self.df[(self.df).columns[i]] = pd.to_numeric(self.df[(self.df).columns[i]])
