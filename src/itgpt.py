import torch
import sys

from src.attention import MultiModalAttention
from src.datastruct import TSdata
from src.activations import get_activation
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
                n_layers_qkv=hparams["n_layers_qkv"], bias=hparams["bias"], output_type=hparams["output_type"], init_random=hparams["init_random"], init_tau=hparams["init_tau"], 
                weight_type=hparams["weight_type"], qk_type=hparams["qk_type"], attention_type=hparams["attention_type"], **kw_args_mlp
            )
            
        if  self.decoder:
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

        the_decoder_input = {m: TSdata(batch[m].data[...,[-1]], batch[m].timeline) for m in batch.keys() if m!= "reference"}
        
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


class ITGPT(torch.nn.Module):
    def __init__(self, hparams):
        super(ITGPT, self).__init__()
        self.hparams = hparams
        self.model = torch.nn.Sequential(*[ItnetBlock(hparams,decoder=i<(hparams["itnet_n_layers"]-1)) for i in range(hparams["itnet_n_layers"])])

    def forward(self, x):
        xhat, z = self.model(x)
        return xhat, z