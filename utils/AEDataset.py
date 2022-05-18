import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

def create_dataset_for_autoencoder(df,features = [],ws=200):
  
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
    out.append((window, window))

  return (out)

class Time_series_Dataset(Dataset):
  
  def __init__(self, dataset_list):
    super().__init__()

    self.dataset_list= dataset_list

  def __len__(self):
    return len(self.dataset_list)

  def __getitem__(self, idx):
    data_window, label = self.dataset_list[idx]

    data_window = data_window.squeeze(1)
    
    if torch.cuda.is_available():
      data_window = data_window.cuda()
      label = label.cuda()

    return data_window, label