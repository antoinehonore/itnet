import torch
import sys

from src.attention import MultiModalAttention
from src.datastruct import TSdata
from src.activations import get_activation
from src.normalization import *
import copy

# A wrapper around MultiModalAttention
class ItnetBlock(torch.nn.Module):
    def __init__(self, hparams, decoder=True):
        super(ItnetBlock, self).__init__()
        self.hparams = hparams
        self.decoder = decoder
        self.activation_function = get_activation(hparams["activation"])()

        kw_args_mlp = dict(activation=hparams["activation"], layernorm=hparams["layernorm"], skipconnections=hparams["skipconnections"], skiptemperature=hparams["skiptemperature"],dropout_p=hparams["dropout_p"])
        
        self.encodeMMA = MultiModalAttention(hparams["modalities_dimension"], anchor_dim=hparams["itnet_anchor_dim"],
                n_layers_qkv=hparams["n_layers_qkv"], bias=hparams["bias"], output_type=hparams["output_type"], 
                n_layers_output=hparams["n_layers_output"], init_random=hparams["init_random"], init_tau=hparams["init_tau"], 
                weight_type=hparams["weight_type"], qk_type=hparams["qk_type"], attention_type=hparams["attention_type"], **kw_args_mlp
            )
        
        if self.decoder:
            decoder_modalities = {mname: dict(in_q=D["out_qk"], in_kv=hparams["itnet_anchor_dim"], out_qk=D["out_qk"], out_v=D["in_kv"]) for mname, D in hparams["modalities_dimension"].items()}

            self.decodeMMA = MultiModalAttention(decoder_modalities, anchor_dim=hparams["itnet_anchor_dim"],
                    n_layers_qkv=hparams["n_layers_qkv"], bias=hparams["bias"], init_random=hparams["init_random"], init_tau=hparams["init_tau"], 
                    weight_type=hparams["weight_type"], qk_type=hparams["qk_type"], attention_type=hparams["attention_type"], **kw_args_mlp
                )

    def __repr__(self):
        return "Encoder({})".format(self.encodeMMA.__repr__()) + ("\nDecoder({})".format(self.decodeMMA.__repr__()) if self.decoder else "")

    def forward(self, args, pool=None, only_last=True):
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

        the_decoder_input = {m: TSdata(batch[m].timeline.unsqueeze(0).unsqueeze(-1), batch[m].timeline, batch[m].idx) for m in batch.keys() if m!= "reference"}
        
        anchor_data = self.encodeMMA(batch)
        
        if self.decoder:
            if self.hparams["itnet_skipconnections"]:
                anchor_data = self.activation_function(anchor_data) + previous_encoded_data 
        
            the_decoder_input["reference"] = TSdata(anchor_data.unsqueeze(1), batch["reference"].timeline,batch["reference"].idx)
            
            yhat = self.decodeMMA(the_decoder_input, mode="decode")
            yhat = {m: TSdata(yhat[m], batch[m].timeline,batch[m].idx) for m in yhat.keys()}
            yhat["reference"] = batch["reference"] 
        else:
            yhat = batch
        return yhat, anchor_data

class Embedding(torch.nn.Module):
    def __init__(self, dimensions):
        super(Embedding, self).__init__()
        self.dimensions = dimensions
        self.embedding_layers = torch.nn.ModuleDict({mname: torch.nn.Linear(dims[0], dims[1]) for mname,dims in dimensions.items()})

    def __repr__(self):
        return "{}x Linear(...)".format(len(self.embedding_layers))

    def forward(self,batch):
        output = {mname: 
                TSdata(self.embedding_layers[mname](batch[mname].data), batch[mname].timeline, batch[mname].idx)
                for mname in self.embedding_layers.keys()
                }
        output["reference"] = batch["reference"]
        return output

class ITGPT(torch.nn.Module):
    def __init__(self, hparams):
        super(ITGPT, self).__init__()
        self.hparams = dict(hparams)
        # (d_in_q, d_in_kv, d_qk, d_out)
        input_dimensions = {mname: (D["in_kv"], int(round(D["in_kv"]*(1+self.hparams["itnet_embedding_dim_p"])))) 
                                    for mname,D in self.hparams["modalities_dimension"].items()
                            }
        
        output_dimensions = {mname: D[::-1] 
                                    for mname, D in input_dimensions.items()
                            }
        
        self.embedding = Embedding(input_dimensions)
        self.output_embedding = Embedding(output_dimensions)
        
        self.output_anchor = torch.nn.Linear(self.hparams["itnet_anchor_dim"], self.hparams["d_out"])
        
        # (d_in_q, d_in_kv, d_qk, d_out)
        blocks_dimensions = {mname: dict(in_q=D["in_q"], in_kv=input_dimensions[mname][1], out_qk=D["out_qk"], out_v=input_dimensions[mname][1]) 
                        for mname, D in self.hparams["modalities_dimension"].items()}

        self.hparams["modalities_dimension"] = blocks_dimensions
        
        self.model = torch.nn.Sequential(*[ItnetBlock(self.hparams, decoder=i<(hparams["itnet_n_layers"]-1)) for i in range(hparams["itnet_n_layers"])])

        self.normalization = self.hparams["normalization"]

        if self.normalization == "batch":
            self.norm_funcs = torch.nn.ModuleDict({mname: torch.nn.BatchNorm2d(dims[1]) for mname,dims in self.hparams["modalities_dimension"].items()})
        elif self.normalization == "log":
            self.norm_funcs = apply_log#torch.nn.ModuleDict({mname: apply_log for mname,dims in hparams["modalities_dimension"].items()})
        elif self.normalization == "domain":
            self.norm_funcs = apply_domain_normalization#torch.nn.ModuleDict({mname: apply_log for mname,dims in hparams["modalities_dimension"].items()})
        elif self.normalization == "domain1":
            self.norm_funcs = apply_domain1_normalization#torch.nn.ModuleDict({mname: apply_log for mname,dims in hparams["modalities_dimension"].items()})
        else:
            raise Exception("Unknown normalization={}".format(self.normalization))
        self.reset_running_slopes()

    def reset_running_slopes(self):
        self.running_slopes = {m: [] for m in self.hparams["modalities_dimension"] if (m != "specs") and (m != "reference")}
        self.training_slopes = {m: None for m in self.hparams["modalities_dimension"] if (m != "specs") and (m != "reference")}

    def normalize(self, batch):
        self.normalized_batch = {m: TSdata(
                        self.apply_norm(m, batch), 
                        batch[m].timeline,
                        batch[m].idx)
            if ((m!="reference") and (m!="specs")) else batch[m]#.clone()
            for m in batch.keys()
        }
        return self.normalized_batch
    
    def forward(self, batch):
        thedata = self.normalize(batch)
        thedata = self.embedding(thedata)
        self.embedded_batch = thedata
        xhat, z = self.model(thedata)
        xhat = self.output_embedding(xhat)
        z = self.output_anchor(z)
        return xhat, z

    def apply_batchnorm(self,m,batch):
        if not (self.norm_funcs is None):
            X = batch[m].data
            xin = X
            if X.shape[2] == 1:
                #Cheat in case T=1: use std=0
                xin = X.expand(-1,-1, 2,-1)
            xout = self.norm_funcs[m](xin.transpose(1,3)).transpose(1,3)
            if X.shape[2] == 1:
                xout = xout[:,:,:1,:]
        else:
            xout = batch[m].data
        return xout

    def compute_slope(self, tsdata):
        """data is (N,H,T,d) and timeline is (N,T)"""
        dx = torch.diff(tsdata.data,dim=-2)
        dt = torch.diff(tsdata.timeline,dim=-1).unsqueeze(1).unsqueeze(-1)

        dt[dt == 0] = torch.nan
        slope =  (dx/dt).nanmean(-2,keepdim=True)
        slope[slope == 0] = 1
        return slope

    def accumulate_slopes(self,m,tsdata):
        slope = self.compute_slope(tsdata)

        if not (slope.isnan().any()):
            self.running_slopes[m].append(slope)

    def correct_slopes(self, m, tsdata):
        if self.training_slopes[m] is None:
            self.training_slopes[m] = sum(self.running_slopes[m])/len(self.running_slopes[m])
        
        slope = self.compute_slope(tsdata)
        if not (slope.isnan().any()):

            corrected_data = tsdata.data*self.training_slopes[m]/slope
        else:
            corrected_data = tsdata.data
        
        return TSdata(corrected_data, tsdata.timeline)

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
    
    def prep_data(self, batch):
        return batch["data"]
    
    def forward(self, batch):
        thefeatures = self.prep_data(batch)
        yhat, z = self.itgpt(thefeatures)
        return yhat, z
