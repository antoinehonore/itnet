import torch
import sys

from src.attention import MultiModalAttention
from src.datastruct import TSdata
from src.activations import get_activation
from src.normalization import *

# A wrapper around MultiModalAttention
class ItnetBlock(torch.nn.Module):
    def __init__(self, hparams, decoder=True):
        super(ItnetBlock, self).__init__()
        self.hparams = hparams
        
        self.decoder = decoder

        self.activation_function = get_activation(hparams["activation"])()

        self.data_augmentation_pdrop = hparams["data_augmentation_pdrop"]
        self.data_augmentation_n = hparams["data_augmentation_n"]
        
        kw_args_mlp = dict(activation=hparams["activation"], layernorm=hparams["layernorm"], skipconnections=hparams["skipconnections"], skiptemperature=hparams["skiptemperature"],dropout_p=hparams["dropout_p"])
        
        self.encodeMMA = MultiModalAttention(hparams["modalities_dimension"],
                n_layers_qkv=hparams["n_layers_qkv"], bias=hparams["bias"], output_type=hparams["output_type"], 
                n_layers_output=hparams["n_layers_output"], init_random=hparams["init_random"], init_tau=hparams["init_tau"], 
                weight_type=hparams["weight_type"], qk_type=hparams["qk_type"], attention_type=hparams["attention_type"], **kw_args_mlp
            )
            
        if self.decoder:
            # 
            # d_in_q, d_in_kv, d_qk, d_out
            decoder_modalities = {mname: (d_in_q, d_out, d_qk, d_in_kv ) for mname, (d_in_q, d_in_kv, d_qk, d_out) in hparams["modalities_dimension"].items()}

            self.decodeMMA = MultiModalAttention(decoder_modalities,
                    n_layers_qkv=hparams["n_layers_qkv"], bias=hparams["bias"], init_random=hparams["init_random"], init_tau=hparams["init_tau"], 
                    weight_type=hparams["weight_type"], qk_type=hparams["qk_type"], attention_type=hparams["attention_type"], **kw_args_mlp
                )
        
    def forward(self, args, pool=None,only_last=True):
        """
        batch is a dictionnary : {"reference":  shape (1,1,T_1,d_1), "m1":  shape (1,1,T_2,d_2), ...}
        """
        
        if isinstance(args,dict):
            batch = args
            previous_encoded_data = 0
        elif isinstance(args,tuple) and len(args)==2:
            batch, previous_encoded_data = args
        else:
            raise Exception("len(args)={} should be 1 or 2.".format(len(args)))

        the_decoder_input = {m: TSdata(batch[m].timeline.unsqueeze(0).unsqueeze(-1), batch[m].timeline) for m in batch.keys() if m!= "reference"}
        
        the_encoded_data = self.encodeMMA(batch)
        
        the_encoded_data = self.activation_function(the_encoded_data)
        
        if self.hparams["itnet_skipconnections"]:
            the_encoded_data = the_encoded_data + previous_encoded_data 
        
        if self.decoder:
            the_decoder_input["reference"] = TSdata(the_encoded_data.unsqueeze(1), batch["reference"].timeline)
            
            yhat = self.decodeMMA(the_decoder_input, mode="decode")
            yhat = {m: TSdata(yhat[m], batch[m].timeline) for m in yhat.keys()}
            yhat["reference"] = batch["reference"] 
        else:
            yhat = batch
        return yhat, the_encoded_data


class Embedding(torch.nn.Module):
    def __init__(self, dimensions):
        super(Embedding, self).__init__()
        self.embedding_layers = torch.nn.ModuleDict({mname: torch.nn.Linear(dims[0], dims[1]) for mname,dims in dimensions.items()})
    
    def forward(self,batch):
        return {mname: 
                    TSdata(self.embedding_layers[mname](batch[mname].data), batch[mname].timeline) if mname != "reference" else batch[mname]
                    for mname in batch.keys()
                    }

class ITGPT(torch.nn.Module):
    def __init__(self, hparams):
        super(ITGPT, self).__init__()
        self.hparams = hparams
        
        input_dimensions = {mname: (d_in_kv, int(round(d_in_kv*(1+hparams["itnet_embedding_dim_p"])))) 
                                    for mname,(d_in_q, d_in_kv, d_qk, d_out) in hparams["modalities_dimension"].items()
                            }
                             
        output_dimensions = {mname: D[::-1] 
                                    for mname, D in input_dimensions.items()
                            }

        self.embedding = Embedding(input_dimensions)
        self.output_embedding = Embedding(output_dimensions)
        self.output_anchor = torch.nn.Linear(hparams["itnet_anchor_dim"],hparams["d_out"])

        dimensions = {mname: (d_in_q, input_dimensions[mname][1], d_qk, hparams["itnet_anchor_dim"]) 
                        for mname, (d_in_q, d_in_kv, d_qk, d_out) in hparams["modalities_dimension"].items()}

        hparams["modalities_dimension"] = dimensions
        
        self.model = torch.nn.Sequential(*[ItnetBlock(hparams, decoder=i<(hparams["itnet_n_layers"]-1)) for i in range(hparams["itnet_n_layers"])])

        self.normalization = hparams["normalization"]

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
    
    def forward(self, batch):
        thedata = {m: TSdata(
                        self.apply_norm(m, batch), 
                        batch[m].timeline)
            if (m!="reference" and m!="specs") else batch[m]#.clone()
            for m in batch.keys()
        }
        thedata = self.embedding(thedata)
        xhat, z = self.model(thedata)
        xhat = self.output_embedding(xhat)
        z = self.output_anchor(z)
        return xhat, z

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
             return self.norm_funcs(m, batch[m].data.clone())
        else:
            raise Exception("Unknown normalization={}".format(self.normalization))

class Predictor(torch.nn.Module):
    def __init__(self, hparams):
        super(Predictor, self).__init__()
        self.hparams = hparams
        self.itgpt = ITGPT(hparams)
    
    def forward(self, batch):
        thefeatures = {}
        thefeatures["reference"] = TSdata(batch["data"]["reference"].T.unsqueeze(0).unsqueeze(0), batch["data"]["reference"])# batch["data"]["reference"]
        thefeatures = {**thefeatures,**{m: TSdata(v.unsqueeze(1), v[..., -1]) for m,v in batch["data"].items() if m!="reference"} }
        
        yhat, z = self.itgpt(thefeatures)
        return yhat, z
