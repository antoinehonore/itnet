
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_size, kernel_size=5, h_dim=12):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(input_size, h_dim, kernel_size)
        self.pool = nn.MaxPool1d(kernel_size)
        self.conv2 = nn.Conv1d(h_dim, h_dim, kernel_size)

    def forward(self, x):
        """x of size (N,C,T)"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv2(x)
        #x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        #_,Cout,T = x.shape
        self.data_out = x
        return x#.reshape(N,T1,Cout,T).transpose(-1,-2)
