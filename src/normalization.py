
from src.compx.mydata import num_variables, cat_variables
import torch

def apply_log(t, eps=1):
    #if (t<0).any():
    #    print("")
    out = (t.sign()*2-1)*((t.abs() + eps).log())
    return out

def apply_domain_normalization(m, t, eps=1e-5):
    if (m in cat_variables.keys()):
        t[...,:-2] = t[...,:-2] / (t[...,:-2].sum(-1).unsqueeze(-1)+eps)
        t = apply_log(t)
    else:
        t = apply_log(t)
    return t

def apply_domain1_normalization(m, t, eps=1e-5):
    # Initial data dim
    #d = (t.shape[-1])#//2
    # Remove the time things
    #t[...,-2:] *=0 # t[...,-2:] * torch.zeros(t[...,-2:].shape,device=t.device,dtype=t.dtype)
    
    # Remove non-normalized data
    #t[...,:d]*=0
    if (m in cat_variables.keys()):
        # Normalize histograms
        t = t / (t.sum(-1).unsqueeze(-1)+eps)
    
    #t = t - t.mean(-2, keepdim=True)
    t = apply_log(t)
    return t
