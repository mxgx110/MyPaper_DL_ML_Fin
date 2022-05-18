from AEDataset import create_dataset_for_autoencoder, Time_series_Dataset
from AEArchitecture import auto_encoder
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def AE_INDICATORS_OSCILLATORS(df,features = [], window_size = 24,
                  train_size = 0.8, batch_size = 32,epochs = 600, lr = 0.0001,
                  conv1_kernel_num = 9, conv1_kernel_size = 7, conv2_kernel_num = 6,
                   conv2_kernel_size = 7,conv3_kernel_num = 3, conv3_kernel_size = 7,
                   maxpool_kernel_size = 2,max_pool_stride = 2, sanity = False):
  
  dataset = create_dataset_for_autoencoder(df, features= features, ws = window_size)

  n = int(train_size * len(dataset))
  m = len(dataset) - n
  start_train = n % batch_size
  end_test = m % batch_size

  train_set = dataset[start_train:n]
  if end_test:
    val_set = dataset[n:-end_test]
  else:
    val_set = dataset[n:]


  train_dataset = Time_series_Dataset(train_set)
  val_dataset = Time_series_Dataset(val_set)

  train_dataloader = DataLoader(train_dataset, batch_size = batch_size)
  val_dataloader = DataLoader(val_dataset, batch_size= batch_size)

  if sanity:
    for data_window, label in train_dataloader:
      break
    torch.manual_seed(42)
    model = auto_encoder(channel_dim = len(features), conv1_kernel_num = conv1_kernel_num, conv1_kernel_size = conv1_kernel_size,
                         conv2_kernel_num = conv2_kernel_num, conv2_kernel_size = conv2_kernel_size, conv3_kernel_num = conv3_kernel_num,
                         conv3_kernel_size = conv3_kernel_size ,maxpool_kernel_size = maxpool_kernel_size, max_pool_stride = max_pool_stride,
                         window_size = window_size)

    model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    model.train()

    data_window = data_window.permute(0,2,1)
    label = label.permute(0,2,1)
    for i in range(20000):

      out = model(data_window)
      loss = criterion(out, label)
      print(f'Loss: {loss}')
  
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    print('\nsanity check is done')
    return (True,True,True)

  else:
    print("Training Process...")
    torch.manual_seed(42)
    model = auto_encoder(channel_dim = len(features), conv1_kernel_num = conv1_kernel_num, conv1_kernel_size = conv1_kernel_size,
                         conv2_kernel_num = conv2_kernel_num, conv2_kernel_size = conv2_kernel_size, conv3_kernel_num = conv3_kernel_num,
                         conv3_kernel_size = conv3_kernel_size ,maxpool_kernel_size = maxpool_kernel_size, max_pool_stride = max_pool_stride,
                         window_size = window_size)

    model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    losses = []
    model.train()
    for i in range(epochs):
      for seq,y_train in train_dataloader:
        seq = seq.permute(0,2,1)
        y_train = y_train.permute(0,2,1)

        optimizer.zero_grad()
        y_pred = model(seq)
        loss = criterion(y_pred, y_train)
  
        loss.backward()
        optimizer.step()
        #scheduler.step()
      losses.append(loss)
      print(f"Epoch {i} Loss: {(loss.item())}")
    plt.plot(losses)
    plt.show()
    print("\nTraining Process has finished")
    print("#############################")
    
    if train_size != 1.0:
      model.eval()
      print("Evaluate the model...")
      with torch.no_grad():
 
        for x,y in val_dataloader:

          x = x.permute(0,2,1)
          y = y.permute(0,2,1)
          y_pred = model(x)
          loss = criterion(y_pred, y)
          print(f'Loss: {loss.item()**0.5}')
      print("\nEvaluation has finished")
      print("#############################")
      print("Visualization...")

      for data_window_, label_ in val_dataloader:
        break

      data_window_ = data_window_.permute(0,2,1)
      label_ = label_.permute(0,2,1)
      data_window_ = torch.Tensor.cpu(data_window_)
      label_ = torch.Tensor.cpu(label_)
      pre = model(data_window_.cuda())
      pre = torch.Tensor.cpu(pre)
      fig=plt.figure(figsize=(8, 4))
      fig.add_subplot(1, 3, 1)
      plt.imshow(data_window_[:,:,:3]) 
      fig.add_subplot(1, 3, 2)
      plt.imshow(label_[:,:,:3])
      fig.add_subplot(1, 3, 3)
      plt.imshow(pre.detach().numpy()[:,:,:3])

    return (model, dataset)