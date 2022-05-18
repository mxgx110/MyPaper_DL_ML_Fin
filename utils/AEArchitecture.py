import torch.nn as nn
import torch.nn.functional as F
class auto_encoder(nn.Module):

  def __init__(self, channel_dim, conv1_kernel_num, conv1_kernel_size, conv2_kernel_num, conv2_kernel_size,
               conv3_kernel_num, conv3_kernel_size ,maxpool_kernel_size,max_pool_stride, window_size):

    super().__init__()

    self.channel_dim = channel_dim
    self.kernel_num_1 = conv1_kernel_num
    self.kernel_size_1 = conv1_kernel_size
    self.kernel_num_2 = conv2_kernel_num
    self.kernel_size_2 = conv2_kernel_size
    self.kernel_num_3 = conv3_kernel_num
    self.kernel_size_3 = conv3_kernel_size
    self.maxpool_kernel_size = maxpool_kernel_size
    self.max_pool_stride = max_pool_stride
    self.window_size = window_size


    #ENCODE

    self.conv1 = nn.Conv1d(self.channel_dim, self.kernel_num_1, self.kernel_size_1, stride=1, padding=int((self.kernel_size_1 - 1)/2))
    self.bn1 = nn.BatchNorm1d(self.kernel_num_1)

    self.conv2 = nn.Conv1d(self.kernel_num_1, self.kernel_num_2, self.kernel_size_2, stride=1, padding=int((self.kernel_size_2 - 1)/2))
    self.bn2 = nn.BatchNorm1d(self.kernel_num_2)

    self.conv3 = nn.Conv1d(self.kernel_num_2, self.kernel_num_3, self.kernel_size_3, stride=1, padding=int((self.kernel_size_3 - 1)/2))
    self.bn3 = nn.BatchNorm1d(self.kernel_num_3)


    #DECODE
   
    self.convB1 = nn.Conv1d(self.kernel_num_3, self.kernel_num_2, self.kernel_size_3, stride=1, padding=int((self.kernel_size_3 - 1)/2))
    self.bnB1 = nn.BatchNorm1d(self.kernel_num_2)

    self.convB2 = nn.Conv1d(self.kernel_num_2, self.kernel_num_1, self.kernel_size_2, stride=1, padding=int((self.kernel_size_2 - 1)/2))
    self.bnB2 = nn.BatchNorm1d(self.kernel_num_1)
 
    self.convB3 = nn.Conv1d(self.kernel_num_1, self.channel_dim, self.kernel_size_1, stride=1, padding=int((self.kernel_size_1 - 1)/2))
    self.bnB3 = nn.BatchNorm1d(self.channel_dim)

    self.convB_final = nn.Conv1d(self.channel_dim, self.channel_dim, self.kernel_size_1, stride=1, padding=int((self.kernel_size_1 - 1)/2))
    self.bnB_final = nn.BatchNorm1d(self.channel_dim)


    #Maxpool and Upsampling

    self.pool = nn.MaxPool1d(kernel_size = self.maxpool_kernel_size, stride = self.max_pool_stride)
    self.up = nn.Upsample(scale_factor=2, mode='nearest')
    

  def forward(self, X):


    #ENCODER

    X = self.bn1(self.conv1(X))
    X = F.relu(X)
    X = self.pool(X)
    
    X = self.bn2(self.conv2(X))
    X = F.relu(X)
    X = self.up(X)

    X = self.conv3(X)
    self.output = X #The output of the AE
    X = self.bn3(X)
    X = F.relu(X)
    X = self.pool(X)

    
    #DECODER
    
    X = self.bnB1(self.convB1(X))
    X = F.relu(X)
    X = self.up(X)
    
    X = self.bnB2(self.convB2(X))
    X = F.relu(X)
    X = self.pool(X)
    
    X = self.bnB3(self.convB3(X))
    X = F.relu(X)
    X = self.up(X)

    X = self.bnB_final(self.convB_final(X))
        
    return X