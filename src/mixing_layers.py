
import torch
from src.mlp import MLP

class Linear(torch.nn.Module):
    def __init__(self,d_in):
        super(Linear,self).__init__()
        self.linear_function = torch.nn.Linear(d_in, d_in, bias=False)

    def forward(self, batch):
        return self.linear_function(batch)

class OutputLayer(torch.nn.Module):
    def __init__(self,d_in, d_out, names, output_type="linear"):
        super(OutputLayer,self).__init__()
        if output_type == "qlinear":
            layer = QLinear
        elif output_type == "linear":
            layer = Linear
        else:
            raise Exception("Unknown output_type={}".format(output_type))

        self.names = sorted(names)
        assert d_in == d_out, "Different input and output dimensions are NYI"
        self.linear_functions =  torch.nn.ModuleDict({mname: layer(d_in) for mname in names})

    def forward(self, batch):
        output = {mname: self.linear_functions[mname](batch[mname]) for mname in self.names}
        yhat = torch.cat(list(output.values()), dim=1)
        # yhat is (N,H,T,d)
        yhat = yhat.sum(1)
        return yhat

class FullOutputLayer(torch.nn.Module):
    def __init__(self,d_in, d_out, names, output_type="fulllinear", n_layers=None,d_qk=None,bias=True, kw_args_mlp={}):
        super(FullOutputLayer,self).__init__()
        self.names = sorted(names)
        #assert d_in == d_out, "Different input and output dimensions are NYI"
        if "linear" in output_type:
            self.function =  torch.nn.Linear(d_in, d_out, bias=bias)# for mname in names})
        elif "mlp" in output_type:
            self.function =  MLP(d_in, [d_qk]*n_layers, d_out, bias=bias, **kw_args_mlp)
        else:
            raise Exception("Unknown output_type={}".format(output_type))

    def forward(self, batch):
        theinput = [batch[mname] for mname in self.names]
        x = torch.cat(theinput, dim=1)
        # x is (N,H,T,d)
        x = x.transpose(1,2).flatten(start_dim=-2,end_dim=-1)
        # x is (N,T,Hd)
        yhat = self.function(x)
        return yhat

class QLinear(torch.nn.Module):
    def __init__(self, d_in):
        super(QLinear,self).__init__()
        self.linear_function = torch.nn.Linear(d_in, d_in, bias=False)

    def forward(self, x):
        Q = torch.linalg.qr(self.linear_function.weight).Q
        yhat = x @ Q.T.to(dtype=x.dtype)
        return yhat

class HouseholderLinear(torch.nn.Module):
    def __init__(self, features, num_reflections=4):
        super(HouseholderLinear, self).__init__()
        self.features = features
        self.num_reflections = num_reflections
        
        # Learnable Householder vectors
        self.vectors = torch.nn.Parameter(torch.randn(num_reflections, features))
        
    def householder_reflection(self, v):
        v = torch.nn.functional.normalize(v, dim=-1)  # Ensure unit vector
        
        vvT = torch.ger(v, v)  # Outer product
        H = self.I - 2 * vvT
        return H

    def construct_orthogonal_matrix(self):
        self.I = torch.eye(self.features, device=self.vectors.device)
        Q = torch.eye(self.features, device=self.vectors.device)
        for i in range(self.num_reflections):
            H = self.householder_reflection(self.vectors[i])
            Q = H @ Q
        return Q

    def forward(self, x):
        W = self.construct_orthogonal_matrix()  # Shape: [features, features]
        return x @ W.T  # Apply linear transformation
