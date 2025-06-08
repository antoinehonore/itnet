import torch
import sys

from src.attention import MultiModalAttention
from src.datastruct import TSdata
from src.compx.mydata import num_variables, cat_variables

class Predictor(torch.nn.Module):
    def __init__(self, hparams):
        super(Predictor, self).__init__()
        self.hparams = hparams
        self.itnet = Itnet(hparams)
    
    def forward(self, batch):
        thefeatures = {}
        thefeatures["reference"] = TSdata(batch["data"]["reference"].T.unsqueeze(0).unsqueeze(0), batch["data"]["reference"])# batch["data"]["reference"]
        thefeatures = {**thefeatures,**{m: TSdata(v.unsqueeze(1), v[..., -1]) for m,v in batch["data"].items() if m!="reference"} }
        
        yhat = self.itnet(thefeatures)
        return yhat


# A wrapper around MultiModalAttention
class Itnet(torch.nn.Module):
    def __init__(self, hparams):
        super(Itnet, self).__init__()
        self.hparams = hparams  
        
        self.data_augmentation_pdrop = hparams["data_augmentation_pdrop"]
        self.data_augmentation_n = hparams["data_augmentation_n"]
        self.normalization = hparams["normalization"]

        kw_args_mlp = dict(activation=hparams["activation"], layernorm=hparams["layernorm"], skipconnections=hparams["skipconnections"], skiptemperature=hparams["skiptemperature"],dropout_p=hparams["dropout_p"])
        if self.normalization == "batch":
            self.norm_funcs = torch.nn.ModuleDict({mname: torch.nn.BatchNorm2d(dims[1]) for mname,dims in hparams["modalities_dimension"].items()})
        elif self.normalization == "log":
            self.norm_funcs = apply_log#torch.nn.ModuleDict({mname: apply_log for mname,dims in hparams["modalities_dimension"].items()})
        elif self.normalization == "domain":
            self.norm_funcs = apply_domain_normalization#torch.nn.ModuleDict({mname: apply_log for mname,dims in hparams["modalities_dimension"].items()})
        elif self.normalization == "domain1":
            self.norm_funcs = apply_domain1_normalization#torch.nn.ModuleDict({mname: apply_log for mname,dims in hparams["modalities_dimension"].items()})

        else:
            raise Exception("Unknown normalization={}".format(self.normalization))
        
        self.MMA = MultiModalAttention(hparams["modalities_dimension"], 
                n_layers_qkv=hparams["n_layers_qkv"], bias=hparams["bias"], output_type=hparams["output_type"],
                init_random=hparams["init_random"], init_tau=hparams["init_tau"], 
                weight_type=hparams["weight_type"], qk_type=hparams["qk_type"], attention_type=hparams["attention_type"], **kw_args_mlp
            )

    def apply_batchnorm(self,m,batch):
        """Cheat in case T=1: use std=0"""
        if not (self.norm_funcs is None):
            X = batch[m].data
            xin = X
            if X.shape[2] == 1:
                xin = X.expand(-1,-1, 2,-1)
            xout = self.norm_funcs[m](xin.transpose(1,3)).transpose(1,3)
            if X.shape[2] == 1:
                xout = xout[:,:,:1,:]
        else:
            xout = batch[m].data
        return xout

    def apply_norm(self, m, batch):
        if self.normalization == "batch":
            return self.apply_batchnorm(m, batch)
        elif self.normalization == "log":
            return self.norm_funcs(batch[m].data)
        elif "domain" in self.normalization:
             return self.norm_funcs(m, batch[m].data)
        else:
            raise Exception("Unknown normalization={}".format(self.normalization))

    def forward(self, batch, pool=None,only_last=True):
        """
        batch is a dictionnary : {"reference":  shape (1,1,T_1,d_1), "m1":  shape (1,1,T_2,d_2), ...}
        """
        thedata = {m: TSdata(
                        self.apply_norm(m, batch), 
                        batch[m].timeline)
                        if (m!="reference" and m!="specs") else batch[m] 
                        for m in batch.keys()
                        }
        yhat = self.MMA( thedata )
        return yhat


