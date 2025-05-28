import torch
import sys

from src.attention import MultiModalAttention
from src.datastruct import TSdata

# A wrapper around MultiModalAttention
class Itnet(torch.nn.Module):
    def __init__(self, hparams):
        super(Itnet, self).__init__()
        self.hparams = hparams  
        
        self.data_augmentation_pdrop = hparams["data_augmentation_pdrop"]
        self.data_augmentation_n = hparams["data_augmentation_n"]
        
        kw_args_mlp = dict(activation=hparams["activation"], layernorm=hparams["layernorm"], skipconnections=hparams["skipconnections"], skiptemperature=hparams["skiptemperature"],dropout_p=hparams["dropout_p"])

        self.MMA = MultiModalAttention(hparams["modalities_dimension"], n_layers=hparams["n_layers"], 
                n_layers_qkv=hparams["n_layers_qkv"], bias=hparams["bias"], output_type=hparams["output_type"],
                init_random=hparams["init_random"], init_tau=hparams["init_tau"], 
                weight_type=hparams["weight_type"], qk_type=hparams["qk_type"], attention_type=hparams["attention_type"], **kw_args_mlp
            )

    def forward(self, batch, pool=None,only_last=True):
        """
        batch is a dictionnary : {"reference":  shape (1,1,T_1,d_1), "m1":  shape (1,1,T_2,d_2), ...}
        """
        #drop(, self.data_augmentation_pdrop, self.data_augmentation_n)
        thedata = {m: batch[m] if m!="reference" else batch[m] for m in batch.keys()}
        yhat = self.MMA( thedata )
        return yhat


