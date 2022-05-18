import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

def create_dataset_for_ohlcv(df,features = [],target_col = 'close',ws=200):

  extra = df[target_col].values
  binary_labels = []
  
  for i in range(len(extra)-1):

    if extra[i] < extra[i+1]:

      binary_labels.append(1.0)

    else:
      binary_labels.append(0.0)

  binary_labels.append(binary_labels[-1])
  container = []
  y_org = []

  for i in range(len(features)):

    container.append(df[features[i]].values.astype(float))
    container[i] = torch.FloatTensor(container[i].reshape(-1,1)).view(-1)

  aux_df = pd.DataFrame(None)
  for i in range(len(features)):

    aux_df[features[i]] = container[i].numpy()

  for i in range(len(aux_df)):

    y_org.append(aux_df.iloc[i,:].values)

  y_org = torch.tensor(y_org)

  out = []
  L = len(y_org)

  for i in range(L-ws):

    window = y_org[i : i+ws]
    label = binary_labels[i+ws : i+ws+1]
    out.append((window, torch.tensor(label)))

  return (out)