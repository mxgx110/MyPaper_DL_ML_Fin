import pandas as pd
import numpy as np
import pywt

def wavelet(train_data, test_data, signal='haar', level=3, mode='periodic'):

  applied_wavelet_df = []
  cols = train_data.columns
  index_1 = train_data.index
  index_2 = test_data.index
  index = []
  index.extend(index_1)
  index.extend(index_2)

  for i in range(len(cols)):

    train = train_data[train_data.columns[i]].values
    test = test_data[test_data.columns[i]].values
    
    history = [x for x in train]
    applied_wavelet_data = []

    coeffs = pywt.wavedec(history,signal,mode,level)
    
    CA3, CD3, CD2, CD1 = coeffs
    CD2 = np.zeros(len(CD2))
    CD3 = np.zeros(len(CD3))
    coeffs_new = [CA3,CD3,CD2,CD1]  
    
    y = pywt.waverec(coeffs_new,signal,mode)

    applied_wavelet_data.extend(y)
    
    

    for t in range(len(test)):

      history.append(test[t])
      coeffs = pywt.wavedec(history[t+1:],signal,mode,level)
      CA3, CD3, CD2, CD1 = coeffs
      CD2 = np.zeros(len(CD2))
      CD3 = np.zeros(len(CD3))
      coeffs_new = [CA3,CD3,CD2,CD1]   
      
      y = pywt.waverec(coeffs_new,signal,mode)

      applied_wavelet_data.append(y[-1])

    applied_wavelet_df.append(applied_wavelet_data)
  
  result = pd.DataFrame(np.array(applied_wavelet_df).T)
  result.columns = cols
  result.index = index
  
  wl_train = result.iloc[:len(train_data),:]
  wl_test = result.iloc[len(train_data):,:]
  
  return (wl_train,wl_test)
