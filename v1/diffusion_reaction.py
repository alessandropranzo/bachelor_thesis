import torch
from torch import nn
from torch.func import functional_call, grad, vmap

from collections import OrderedDict
from typing import Callable

from utils import tuple_2_dict
from pdes import inversed_k, inversed_u

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
        layer_list.append(
            ("layer_%d" % (self.depth - 1), nn.Linear(layers[-2], layers[-1]))
        )

        layer_dict = OrderedDict(layer_list)

        self.layers = nn.Sequential(layer_dict)

    def forward(self, x):
        out = self.layers(x)
        return out
    

def make_fwd_inv_fn(model: nn.Module):

    def fn(x: torch.Tensor, params: dict[str, nn.Parameter] | tuple[nn.Parameter]) -> Callable:
        if isinstance(params, tuple):
            params_dict = tuple_2_dict(model, params)
        else:
            params_dict = params  

        return functional_call(model, params_dict, (x, )).squeeze()
    
    return fn

def make_diff_react_loss(fn: Callable) -> Callable:
    
    #This is the standard forward function
    def u(x: torch.Tensor, params: dict[str, nn.Parameter] | tuple[nn.Parameter]) -> Callable:
        return fn(x, params)[:, 0].squeeze() #Take the first entry
    
    #This is the target inverse function (related to the raction coefficient)
    def k(x: torch.Tensor, params: dict[str, nn.Parameter] | tuple[nn.Parameter]) -> Callable:
        return fn(x, params)[:, 1].squeeze() #Take the second entry
    
    #Defining the second derivative w.r.t. x
    def d2udx2(x: torch.Tensor, params: torch.Tensor):
        #Let's define the differentiable part of the the function u(x)
        def u_diff(x, params):
            return fn(x, params)[0]
        #Defining the first derivative w.r.t. x (note that it is not vmapped since it will be used by the second derivative)
        def dudx(x: torch.Tensor, params: torch.Tensor):
            return grad(u_diff)(x, params).squeeze()
        return vmap(grad(dudx), in_dims=(0, None))(x, params).squeeze()
    
    def diff_react_loss(x: torch.Tensor, params: torch.Tensor):
        loss = nn.MSELoss()
        #Data Loss !!!!!!! Not sure that k is needed here, shouldn't we learn starting from some values?
        u_value = u(x, params)
        real_u_value = inversed_u(x)
        k_value = k(x, params)
        real_k_value = inversed_k(x)
        data_loss = loss(u_value, real_u_value) + loss(k_value, real_k_value)
        #Physics Loss (here lambda = 0.01)
        f_value = torch.sin(2 * torch.pi * x).squeeze() + k(x, params) * u(x, params) - 0.01 * d2udx2(x, params)
        physics_loss = loss(f_value, torch.zeros_like(f_value))
        return data_loss + physics_loss

    return diff_react_loss