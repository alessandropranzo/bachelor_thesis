import torch
from torch import nn
from torch.func import functional_call, grad, vmap

from collections import OrderedDict
from typing import Callable

from utils import tuple_2_dict
from pdes import inversed_diffusion_reaction

class DiffReactNN(nn.Module):
    
    def __init__(self, layers: list = [1, 5, 5, 5, 5, 2], act: nn.Module = nn.Tanh) -> None:
        super().__init__()

        self.depth = len(layers) - 1
        self.act = act

        layer_list = list()

        for i in range(self.depth - 1):
            #Adding the linear layer
            layer_list.append(
                ("layer_%d" % i, nn.Linear(layers[i], layers[i+1]))
            )
            #Adding the activation function
            layer_list.append(
                ("activation_%d" % i, self.act())
            )

        #Appending the last layer without the activation function
        layer_list.append("layer_%d" % (self.depth - 1), nn.Linear(layers[-2], layers[-1]))

        layer_dict = OrderedDict(layer_list)

        self.layers = nn.Sequential(layer_dict)

    def forward(self, x):
        out = self.layers(x)
        return out
    

def make_inverse_fn(model: nn.Module):

    def fn(x: torch.Tensor, params: dict[str, nn.Parameter] | tuple[nn.Parameter]) -> Callable:
        if isinstance(params, tuple):
            params_dict = tuple_2_dict(model, params)
        else:
            params_dict = params  

        return functional_call(model, params_dict, (x)) 
    
    return fn