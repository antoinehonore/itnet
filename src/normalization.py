
from src.compx.mydata import num_variables, cat_variables

def apply_log(t, eps=1):
    return (t + eps).log()

def apply_domain_normalization(m, t, eps=1e-5):
    if (m in cat_variables.keys()):
        t[...,:-2] = t[...,:-2] / (t[...,:-2].sum(-1).unsqueeze(-1)+eps)
        t = apply_log(t)
    else:
        t = apply_log(t)
    return t

def apply_domain1_normalization(m, t, eps=1e-5):
    # Remove the time things
    t[...,-2:] *=0 # t[...,-2:] * torch.zeros(t[...,-2:].shape,device=t.device,dtype=t.dtype)
    if (m in cat_variables.keys()):
        # Normalize histograms
        t[...,:-2] = t[...,:-2] / (t[...,:-2].sum(-1).unsqueeze(-1)+eps)
    #else:
    t = apply_log(t)
    return t
