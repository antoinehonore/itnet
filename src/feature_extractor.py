
import torch
from src.cnn import CNN

class ModalityFeatureExtractor(torch.nn.Module):
    def __init__(self, d,  h_dim=12,kernel_size=10):
        super(ModalityFeatureExtractor,self).__init__()
        self.cnn = CNN(d, kernel_size=kernel_size, h_dim=h_dim)
        self.model = torch.nn.LSTM(h_dim, h_dim, batch_first=True)
        
    def forward(self, x):
        N,T1,T2,C = x.shape
        #x.reshape(N*T1,T2,C).transpose(-1,-2)
        
        out = self.cnn(x.reshape(N*T1,T2,C).transpose(-1,-2)).transpose(-1,-2)#.reshape(N*T1,T2,Cout)
        h, (h_n, c_n) = self.model(out)
        h_n = h_n[-1]
        _,Cout = h_n.shape

        return h_n.reshape(N,T1,Cout)
    
class FeatureExtractor(torch.nn.Module):
    def __init__(self, dimensions, kernel_size=10, h_dim=12):
        super(FeatureExtractor,self).__init__()
        self.dimensions = dimensions
        #self.timelines = timelines
        self.modality_models = torch.nn.ModuleDict({m: ModalityFeatureExtractor(d, h_dim=h_dim, kernel_size=kernel_size) for m,d in self.dimensions.items()})
    
    def forward(self, batch):
        features = {}
        for m, thedata in batch["calX"].items():
            h_n = self.modality_models[m](thedata)
            features[m] = torch.cat([h_n, batch["timelines"][m].diff(0), batch["timelines"][m]],dim=-1)
        return features
