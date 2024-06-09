import torch
from torch import nn
from torch.func import functional_call, grad, vmap

from collections import OrderedDict
from typing import Callable

from utils import tuple_2_dict
from pdes import diffusion

class DiffusionNN(nn.Module):
    
    def __init__(self, layers: list = [2, 5, 5, 5, 5, 1], act: nn.Module = nn.Tanh) -> None:
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
    

def make_forward_fn(model: nn.Module):
    #Here x and t are stacked on the same tensor
    def u(x_t: torch.Tensor, params: dict[str, nn.Parameter] | tuple[nn.Parameter]) -> Callable:
        if isinstance(params, tuple):
            params_dict = tuple_2_dict(model, params)
        else:
            params_dict = params  

        return functional_call(model, params_dict, (x_t, )).squeeze() 
    
    return u


def make_diffusion_loss(u: Callable) -> Callable:
    #Gradient of the function
    grad_u = grad(u)

    #Defining the first derivative w.r.t. t
    def dudt(x_t: torch.Tensor, params: torch.Tensor):
        return vmap(grad_u, in_dims=(0, None))(x_t, params)[:, 1:].squeeze()
    
    #Defining the second derivative w.r.t. x
    def d2udx2(x_t: torch.Tensor, params: torch.Tensor):
        #Defining the first derivative w.r.t. x (note that it is not vmapped since it will be used by the second derivative)
        def dudx(x_t: torch.Tensor, params: torch.Tensor):
            return grad_u(x_t, params)[0]
        return vmap(grad(dudx), in_dims=(0, None))(x_t, params)[:, :1].squeeze()

    #Here data loss and physics loss, share the same inputs, while the boundary loss is adjusted to the boundary conditions 
    def diffusion_loss(x:torch.Tensor, t: torch.Tensor, params: torch.Tensor):
        x_t = torch.stack((x, t), dim = 1)

        loss = nn.MSELoss()
        #Data Loss
        u_value = u(x_t, params)
        real_value = diffusion(x, t)
        data_loss = loss(u_value, real_value)

        #Physics Loss
        f_value = dudt(x_t, params) - d2udx2(x_t, params) - torch.exp(-t) * (- torch.sin(torch.pi * x) + (torch.pi**2) * torch.sin(torch.pi * x))
        phy_loss = loss(f_value , torch.zeros_like(f_value))

        #Boundary Losses
        bound1_x_t = torch.stack((x, torch.zeros_like(x)), dim = 1)
        bound2_x_t = torch.stack((torch.ones_like(t), t), dim = 1)
        bound3_x_t = torch.stack((- torch.ones_like(t), t), dim = 1)

        bound1_loss = loss(u(bound1_x_t, params), torch.sin(torch.pi * x)) #u(x,0) = sin(pi*x)
        bound2_loss = loss(u(bound2_x_t, params), torch.zeros_like(t)) #u(1, t) = 0
        bound3_loss = loss(u(bound3_x_t, params), torch.zeros_like(t)) #u(-1, t) = 0

        #Add all the losses and return their values 
        return data_loss + 100 * phy_loss + 200 * bound1_loss + 200 * bound2_loss + 200 * bound3_loss
    
    return diffusion_loss