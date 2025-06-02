
import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_length=1000.0):
        super(PositionalEncoding, self).__init__()
        self.max_seq_length = max_seq_length
        self.d_model = d_model

    def forward(self, x):
        """x is a time vector of size (N,T). 
        Returns a (N,T,d_model) embedding of the time vector"""
        pe = torch.zeros(x.shape[0], x.shape[1], self.d_model, device=x.device)

        div_term = torch.exp(torch.arange(0, self.d_model, 2,device=x.device).float() * -(math.log(self.max_seq_length) / self.d_model))
        pe[:, :, 0::2] = torch.sin(x.unsqueeze(-1) * div_term.unsqueeze(0).unsqueeze(0))
        pe[:, :, 1::2] = torch.cos(x.unsqueeze(-1) * div_term.unsqueeze(0).unsqueeze(0))
        return pe
