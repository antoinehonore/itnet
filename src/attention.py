
import torch
from src.mlp import MLP
from src.mixing_layers import OutputLayer, QLinear, Linear, FullOutputLayer
from src.positional_encoding import PositionalEncoding
from fast_transformers.cross_causal_product import cross_causal_dot_product
from src.datastruct import TSdata
from functools import partial
import math

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

def add_one(x, dim=1):
    out = torch.cat([x,torch.ones(x.shape[:-1]+(1,),device=x.device)],dim=-1)
    if dim == 0:
        out = torch.cat([torch.ones(x.shape[:-1]+(1,),device=x.device),-x],dim=-1)
    return out


class MultiModalAttention(torch.nn.Module):
    def __init__(self, dimensions, bias=True, output_type=None,  n_layers_qkv = 0, n_layers_output=0,
                 qk_type="time", init_random=False, init_tau=1, weight_type="gaussian", attention_type="vanilla", **kw_args_mlp):
        super(MultiModalAttention, self).__init__()
        self.dimensions = dimensions
        #n_layers_qkv=None, 
        self.feature_map_q = add_one
        self.qk_type=qk_type
        d_q_in, _, d_qk, _ = list(self.dimensions.values())[0]
        self.d_qk = d_qk

        self.W_Q = MLP(d_q_in,  [d_qk]*n_layers_qkv, d_qk, bias=False, **kw_args_mlp)

        self.uni_modal_attention = torch.nn.ModuleDict({mname: UniModalAttention(d_q_in, d_kv_in, d_qk, d_v, n_layers_qkv, qk_type,
                                attention_type=attention_type, init_random=init_random, init_tau=init_tau, weight_type=weight_type,
                                name=mname,**kw_args_mlp)
                                for mname, (d_q_in, d_kv_in, d_qk, d_v) in self.dimensions.items() if mname != "reference"})
        
        self.output_layer = None
        if not (output_type is None):
            output_d_in = [l[-1] for l in self.dimensions.values()]

            if not ("full" in output_type):
                output_d_in = output_d_in[0]
                self.output_layer = OutputLayer(output_d_in, output_d_in, list(self.uni_modal_attention.keys()), output_type=output_type)
            else:
                self.output_layer = FullOutputLayer(sum(output_d_in), output_d_in[0], list(self.uni_modal_attention.keys()),
                                    output_type=output_type, n_layers=n_layers_output,d_qk=d_qk, kw_args_mlp=kw_args_mlp)

    def compute_Q(self, data_q):
        timeline = data_q.timeline

        #Q = self.W_Q(data_q.data)

        if self.qk_type == "time":
            Q = self.W_Q(data_q.data)
            Q = self.feature_map_q(Q)
        
        elif self.qk_type == "data+PE":
            t1_pe = PositionalEncoding(self.d_qk)(timeline)
            Q = t1_pe
        
        else:
           raise Exception("Unknown qk_type={}".format(self.qk_type))
        return TSdata(Q, timeline)

    def forward(self, batch, mode="encode"):
        """
            Batch is a dictionnary : {"reference": shape (N, 1, T_1, d_1), "m1": shape (N,1,T_2,d_2), ...}
        """

        data_q = batch["reference"]
        data_q = self.compute_Q(data_q)

        if mode == "encode":
            results    = {mname: uni_modal.forward(data_q, batch[mname]) for mname, uni_modal in self.uni_modal_attention.items()}
            norms      = {mname: r.detach().square().sum(-1).unsqueeze(-1) for mname, r in results.items()}
            tot_norm   = torch.cat(list(norms.values()),dim=-1).sum(-1).unsqueeze(-1)
            self.norms = {k: 100 * v / tot_norm for k,v in norms.items()}
            
        elif mode == "decode":
            results = {mname: uni_modal.forward(batch[mname],data_q) for mname, uni_modal in self.uni_modal_attention.items()}
        else:
            raise Exception("Unknown MMA mode={}".format(mode))

        yhat = results
        if not (self.output_layer is None):
            yhat = self.output_layer(results)
        
        return yhat

class UniModalAttention(torch.nn.Module):
    def __init__(self, d_q_in, d_kv_in, d_qk, d_v, n_layers_qkv, qk_type, bias=True,  init_random=False, init_tau=1, 
        weight_type="gaussian", attention_type="vanilla",name="default",**kw_args_mlp
    ):
        #activation="relu", layernorm=False, skipconnections=False, skiptemperature=False, dropout_p=0,
        super(UniModalAttention,self).__init__()
        self.name = name
        self.feature_map_k = partial(add_one, dim=0)
        self.qk_type = qk_type
        self.d_qk = d_qk
        self.weight_type = weight_type
        self.dropout = torch.nn.Dropout(kw_args_mlp["dropout_p"])

        if attention_type == "vanilla":
            self.causal_attn_func = self.causal_scaled_dot_product_attention
        elif attention_type == "linear":
            self.causal_attn_func = self.causal_scaled_linear_attention
        else:
            raise Exception("Unknown attention_type={}".format(attention_type))
        self.W_K = MLP(d_kv_in, [d_qk]*n_layers_qkv, d_qk, bias=bias,  **kw_args_mlp)
        self.W_V = MLP(d_kv_in, [d_qk]*n_layers_qkv, d_v,  bias=bias,  **kw_args_mlp)

        self.attn_matrices = None
        temperature_init = init_tau
        self.history_temperature = torch.nn.Parameter(torch.tensor(math.log(temperature_init), dtype=torch.float32),requires_grad=True) 
        if "PE" in self.qk_type:
            self.position_encoding = PositionalEncoding(self.d_qk)
    
    def forward(self, data_q, data_kv):
        Q, q_timeline = data_q.data, data_q.timeline

        K, V, kv_timeline = self.compute_KV(data_kv)

        out = self.causal_attn_func(Q, K, V, q_timeline, kv_timeline)
        
        if out.isnan().any():
            raise Exception("{} contains NANs !!!".format(self.name))
        return out
    

    def compute_KV(self, data_kv):
        timeline = data_kv.timeline
        # If all the data should be used to compute Q and K (i.e. the attention weights)
        if "data" in self.qk_type:
            input_keys = data_kv.data
            K = self.W_K(input_keys)
            if "PE" in self.qk_type:
                #timeline = 
                K = K + self.position_encoding(timeline)
            
            input_values = data_kv.data

        # If we only use the timestamps to compute thte attention weights
        elif self.qk_type == "time":
            input_keys = data_kv.data[..., [-1]]

            K = self.feature_map_k(self.W_K(input_keys))
        
            input_values = data_kv.data[...,:-2]

        if not (self.W_V is None):
            V = self.W_V(input_values)
        else:
            V = input_values
        return K,V, timeline

    def causal_scaled_linear_attention(self, Q, K, V, t1, t2,  eps=1e-6):
        y = cross_causal_dot_product(torch.tanh(Q), torch.tanh(K), V, t1[0], t2[0])
        self.attn_matrices = torch.zeros(1,1,2,2)
        return y

    def causal_scaled_dot_product_attention(self, Q, K, V, t1, t2,  eps=1e-6):
        tau = self.history_temperature.exp()
        
        A = self.get_weights(Q, K, t1, t2, tau=tau)
        #remove_idx = A.isnan().sum(-1) == A.shape[-1]
        #if remove_idx.any():
        #    print("")
        self.attn_matrices = A.detach()
        # Were whole rows are NaNs, there was no data prior to the required prediction time. Replace weights with zeros.
        #V = V[:, :, keep_idx[0,0,:],:]
        #[:,:,keep_idx[0,0,:],:]
        y = torch.einsum('nhtl,nhlc->nhtc', self.dropout(A), V)
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

        elif self.weight_type == "vanilla":
            self.A = torch.einsum('nhtc,nhlc->nhtl', Q, K) / math.sqrt(K.shape[-1])
        else:
            raise Exception("Unknown weight_type="+self.weight_type)
        
        mask = t1.unsqueeze(-1) >= t2.unsqueeze(1)
        self.A = (self.A* mask).softmax(-1) *mask
        return self.A
