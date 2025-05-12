import torch
from src.mlp import MLP
from functools import partial
import math
import sys

from fast_transformers.cross_causal_product import cross_causal_dot_product

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

def add_one(x, dim=1):
    out = torch.cat([x,torch.ones(x.shape[:-1]+(1,),device=x.device)],dim=-1)
    if dim == 0:
        out = torch.cat([torch.ones(x.shape[:-1]+(1,),device=x.device),-x],dim=-1)
    return out

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_length=100.0):
        super(PositionalEncoding, self).__init__()
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        #self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x, device="cpu"):
        """x is a time vector of size (N,T). 
        Returns a (N,T,d_model) embedding of the time vector"""
        pe = torch.zeros(x.shape[0], x.shape[1], self.d_model,device=device)

        div_term = torch.exp(torch.arange(0, self.d_model, 2,device=device).float() * -(math.log(self.max_seq_length) / self.d_model))
        pe[:, :, 0::2] = torch.sin(x.unsqueeze(-1) * div_term.unsqueeze(0).unsqueeze(0))
        pe[:, :, 1::2] = torch.cos(x.unsqueeze(-1) * div_term.unsqueeze(0).unsqueeze(0))
        return pe


class UniModalAttention(torch.nn.Module):
    def __init__(self, d_in, d_qk, d_v, n_layers_qk, qk_type, bias=True, activation="relu", layernorm=False, 
        skipconnections=False, skiptemperature=False, init_random=False, init_tau=1, 
        weight_type="gaussian", dropout_p=0, attention_type="vanilla"
    ):
        super(UniModalAttention,self).__init__()

        self.feature_map_k = partial(add_one, dim=0)
        self.qk_type = qk_type
        self.d_qk = d_qk
        self.weight_type = weight_type

        if attention_type == "vanilla":
            self.causal_attn_func = self.causal_scaled_dot_product_attention
        elif attention_type == "linear":
            self.causal_attn_func = self.causal_scaled_linear_attention
        else:
            raise Exception("Unknown attention_type={}".format(attention_type))

        self.W_K = MLP(d_in, [d_qk]*n_layers_qk, d_qk, activation, bias=bias, dropout_p=dropout_p,
                                layernorm=layernorm, skipconnections=skipconnections, skiptemperature=skiptemperature) 
        
        self.W_V = MLP(d_in, [d_qk]*n_layers_qk, d_v, activation, bias=bias, dropout_p=dropout_p,
                                layernorm=layernorm, skipconnections=skipconnections, skiptemperature=skiptemperature) 
        self.attn_matrices = None
        temperature_init = init_tau
        self.history_temperature = torch.nn.Parameter(torch.tensor(math.log(temperature_init), dtype=torch.float32),requires_grad=True) 
        

    def forward(self, X, t1=None, Q=None):

        t2 = X[:, 0, :, -1]

        # If all the data should be used to compute Q and K (i.e. the attention weights)
        if "data" in self.qk_type:
            input_keys = X
            K = self.W_K(input_keys)
            if "PE" in self.qk_type:
                t2_pe = PositionalEncoding(self.d_qk)(t2)
                K = K + t2_pe
            
            input_values = X

        # If we only use the timestamps to compute thte attention weights
        elif self.qk_type == "time":
            input_keys = X[..., [-1]]

            K = self.feature_map_k(self.W_K(input_keys))
        
            input_values = X[...,:-2]

        if not (self.W_V is None):
            V = self.W_V(input_values)
        else:
            V = input_values
        
        out = self.causal_attn_func(Q, K, V, t1, t2)
        
        if out.isnan().any():
            print(modality_name, "NANs !!!")
            sys.exit(1)
        return out
        
    def causal_scaled_linear_attention(self, Q, K, V, t1, t2,  eps=1e-6):
        y = cross_causal_dot_product(torch.tanh(Q), torch.tanh(K), V, t1[0], t2[0])
        self.attn_matrices = torch.zeros(1,1,2,2)
        return y

    def causal_scaled_dot_product_attention(self, Q, K, V, t1, t2,  eps=1e-6):
        tau = self.history_temperature.exp()
        
        A = self.get_weights(Q, K, t1, t2, tau=tau)
        
        self.attn_matrices = A.detach()
        # where whole rows are NaNs, there was no data prior to the required prediction time. Replace weights with zeros. 
        y = torch.einsum('nhtl,nhlc->nhtc', A, V)
        return y
    
    def get_weights(self, Q, K, t1, t2, tau = 1):
        N, H, T, C = Q.shape
        _, _, L, _ = K.shape

        if self.weight_type=="gaussian":
            self.A = torch.einsum('nhtc,nhlc->nhtl', Q, K).square()
            self.A = (-self.A/tau)

        elif self.weight_type=="uniform":
            self.A = torch.ones(N,H,T,L, dtype=Q.dtype, device=Q.device)
        
        elif self.weight_type=="linear":
            self.A = torch.einsum('nhtc,nhlc->nhtl', Q, K).abs()
            self.A = (-self.A/tau)

        attn_bias = torch.zeros(N,H,T,L, dtype=Q.dtype, device=self.A.device)
        attn_bias.masked_fill_((t1.unsqueeze(-1) < t2.unsqueeze(1)).unsqueeze(1), float("-inf"))
        attn_bias.to(Q.dtype)
        self.A += attn_bias
        self.A = self.A.softmax(-1)
        all_nans_idx = self.A.isnan().sum(-1) == self.A.shape[-1]
        if all_nans_idx.any():
            self.A[all_nans_idx] = 0
        return self.A.detach()


class MultiModalAttention(torch.nn.Module):
    def __init__(self, modalities_dimension, d_out, d_qk,  L=1, n_layers_qk=None,bias=True,
                 n_layers=0, activation="relu", 
                 layernorm=False, skipconnections=False, skiptemperature=False, 
                 qk_type="time", init_random=False, init_tau=1, weight_type="gaussian", dropout_p=0,attention_type="vanilla"):
        super(MultiModalAttention,self).__init__()
        self.M = len(modalities_dimension["k"])
        self.d_v = d_out
        self.d_out = d_out
        self.weight_type = weight_type
        self.qk_type = qk_type
        self.attention_type = attention_type

        self.modalities_dimension = modalities_dimension
        self.d_qk = d_qk
        self.feature_map_q = add_one
        self.n_layers_qk = n_layers_qk if not (n_layers_qk is None) else n_layers
        self.init_random = init_random

        self.d_in_q = self.modalities_dimension["q"]
        
        self.uni_modal_models = torch.nn.ModuleDict({mname: UniModalAttention(d_in, d_qk, self.d_v, self.n_layers_qk, qk_type,
                                 activation=activation, layernorm=layernorm, skipconnections=skipconnections, skiptemperature=skiptemperature,
                                attention_type=attention_type, init_random=init_random,init_tau=init_tau,weight_type=weight_type,
                                dropout_p=dropout_p,)
                                    for mname,d_in in self.modalities_dimension["k"].items() if mname != "reference"})
        
        self.W_Q = MLP(self.d_in_q, [self.d_qk]*self.n_layers_qk, self.d_qk, activation, bias=False,dropout_p=dropout_p,
                    layernorm=layernorm, skipconnections=skipconnections, skiptemperature=skiptemperature)
        
        if not self.init_random:
            self.W_Q.linear_layers[0].weight = torch.nn.Parameter(torch.tensor([[1.]],dtype=torch.float32),requires_grad=False)
        
        self.W_out = MLP(self.M*self.d_v, [self.M*self.d_v]*n_layers, d_out, activation, bias=bias,dropout_p=dropout_p,
                                            layernorm=layernorm, skipconnections=skipconnections, skiptemperature=skiptemperature) 
        
    def forward(self, batch, pool=None):
        """
            batch is a dictionnary : {"reference":  shape (N, 1, T_1, d_1), "m1":  shape (N,1,T_2,d_2), ...}
        """
        # keep the timeline intact
        t1 = batch["reference"][:, 0, :, -1]

        Q = self.W_Q(batch["reference"])
        
        if self.qk_type == "time":
            Q = self.feature_map_q(Q)
        
        elif self.qk_type == "data+PE":
            t1_pe = PositionalEncoding(self.d_qk)(t1)
            Q = Q + t1_pe
        else:
            raise Exception("Unknown qk_type={}".format(self.qk_type))

        # Compute individual modality predictions sequentially
        results = [uni_modal.forward(batch[mname], t1=t1, Q=Q) for mname, uni_modal in self.uni_modal_models.items()]

        # Concatenate on the head dimension
        Zout = torch.cat(results, dim=1)

        if not (self.W_out is None):
            # Flatten all the heads
            Zout = Zout.transpose(1,2).flatten(start_dim=2,end_dim=3)
            yhat = self.W_out(Zout)
        else:
            yhat = Zout.mean(1)
        return yhat


# A wrapper around MultiModalAttention
class FusionAttn(torch.nn.Module):
    def __init__(self, hparams):
        super(FusionAttn, self).__init__()
        self.hparams = hparams  
        self.M = len(hparams["modalities_dimension"])
        self.d_out = hparams["d_out"] #if "d_out" in hparams.keys() else hparams["h"] + hparams["h"]
        self.data_augmentation_pdrop = hparams["data_augmentation_pdrop"]
        self.data_augmentation_n = hparams["data_augmentation_n"]
        
        if not ("init_tau" in hparams.keys()):
            hparams["init_tau"] = 1
        
        self.modalities_dimension = hparams["modalities_dimension"]
        self.d_qk = hparams["d_qk"]
        
        self.pool = None
       
        self.estimate_fusion = MultiModalAttention(self.modalities_dimension, self.d_out, 
                d_qk=self.d_qk, n_layers=hparams["n_layers"], activation=hparams["activation"],
                n_layers_qk=hparams["n_layers_qk"], bias=hparams["bias"], 
                init_random=hparams["init_random"], init_tau=hparams["init_tau"], 
                weight_type=hparams["weight_type"], dropout_p=hparams["dropout_p"],
                qk_type=hparams["qk_type"],attention_type=hparams["attention_type"]
            )

    def forward(self, batch, pool=None,only_last=True):
        """
        batch is a dictionnary : {"reference":  shape (1,1,T_1,d_1), "m1":  shape (1,1,T_2,d_2), ...}
        """

        thedata = {m: drop(batch[m], self.data_augmentation_pdrop, self.data_augmentation_n) if m!="reference" else batch[m] for m in batch.keys()}
        yhat = self.estimate_fusion( thedata )
        return yhat

def drop(x, p, n):
    """Create n copies of x with a fraction p of time samples randomly dropped"""
    xout = x
    if p>0:
        T = x.shape[2]
        n_keep = max([1,int(T*(1-p))])
        xout = torch.cat([x[:,:,torch.randperm(T)[:n_keep],:] for _ in range(n)])
    return xout