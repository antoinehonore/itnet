import torch
from src.activations import get_activation


class LinearResidualBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, activation="relu", dropout_p=0, layernorm=False,skipconnections=False,skiptemperature=False,bias=True):
        super(LinearResidualBlock,self).__init__()
        self.linear_function = torch.nn.Linear(in_dim, out_dim,bias=bias)
        self.dropout_function = torch.nn.Dropout(dropout_p)
        self.activation_function = get_activation(activation)()
        
        self.layernorm = layernorm
        self.skipconnections = skipconnections
        self.skiptemperature = skiptemperature
        self.skiptemperature_params = torch.nn.Parameter(torch.zeros(1), requires_grad=False) 

        if self.skiptemperature:
            self.skiptemperature_params.requires_grad_(True)
        
        if self.skipconnections:
            self.linear_layer_skip = torch.nn.Linear(in_dim, out_dim,bias=bias)
        
    def forward(self, x):
        """
            x = dropout(x)
            y = W1(x)
            y = norm(y)
            y = activation(y)
            y = W2(x) + y
        """
        x = self.dropout_function(x)
        y = self.linear_function(x)

        if self.layernorm:
            y = torch.nn.functional.layer_norm(y, normalized_shape=y.shape[1:])
        
        y = self.activation_function(y)

        if self.skipconnections:
            tau = torch.nn.functional.sigmoid(self.skiptemperature_params)
            y = tau*self.linear_layer_skip(x) + (1-tau)*y
        return y

class MLP(torch.nn.Module):
    def __init__(self, input_size, layers_sizes, output_size, activation="relu", dropout_p=0, layernorm=False, skipconnections=False, skiptemperature=False,bias=True):
        super(MLP,self).__init__()
        all_params = dict(dropout_p=dropout_p, layernorm=layernorm, skipconnections=skipconnections, skiptemperature=skiptemperature,bias=bias)
        if len(layers_sizes)>0:
            layers_sizes = [input_size] + layers_sizes + [output_size]

            self.layers = []
            for i in range(len(layers_sizes)-1):
                thelayer = LinearResidualBlock(layers_sizes[i], layers_sizes[i+1], activation, **all_params)
                self.layers.append(thelayer)
        else:
            self.layers = [torch.nn.Linear(input_size,output_size,bias=bias)]
        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x

