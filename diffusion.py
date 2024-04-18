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
        layer_list.append("layer_%d" % (self.depth - 1), nn.Linear(layers[-2], layers[-1]))

        layer_dict = OrderedDict(layer_list)

        self.layers = nn.Sequential(layer_dict)

    def forward(self, x):
        out = self.layers(x)
        return out
    

def make_forward_fn(model: nn.Module):

    def fn(x: torch.Tensor, t: torch.Tensor, params: dict[str, nn.Parameter] | tuple[nn.Parameter]) -> Callable:
        if isinstance(params, tuple):
            params_dict = tuple_2_dict(model, params)
        else:
            params_dict = params  

        return functional_call(model, params_dict, (x, t)) 
    
    return fn


#   USE VMAP ON THE GRADIENTS TO SUPPORT BATCHING and define the in_dim parameter
def make_diffusion_loss(u: Callable) -> Callable:
    #First derivative with respect to time t
    dudt = grad(u, 1)
    dudt = vmap(dudt)
    #First derivative with respect to position x
    dudx = grad(u, 0)
    #Second derivative with respect to position x
    d2udx2 = grad(dudx, 0)
    d2udx2 = vmap(d2udx2)

    #Here data loss and physics loss, share the same inputs, while 
    def diffusion_loss(x: torch.Tensor, t: torch.Tensor, params: torch.Tensor):
        loss = nn.MSELoss()

        #Data Loss DO WE NEED THIS OR NOT?????????????????????????????????????????????????????????
        u_value = u(x, t, params)
        real_value = diffusion(x, t)
        data_loss = loss(u_value, real_value)

        #Physics Loss
        f_value = dudt(x, t, params) - d2udx2(x, t, params) - torch.exp(-t) * (- torch.sin(torch.pi * x) + torch.pi**2 * torch.sin(torch.pi * x))
        phy_loss = loss(f_value , torch.zeros_like(f_value))

        #Boundary Losses
        bound1_t_0 = torch.zeros_like(x)
        bound2_x_0 = torch.ones_like(t)
        bound3_x_0 = - bound2_x_0

        bound1_loss = loss(u(x, bound1_t_0, params), torch.sin(torch.pi * x)) #u(x,0) = sin(pi*x)
        bound2_loss = loss(u(bound2_x_0, t), torch.zeros_like(t)) #u(1, t) = 0
        bound3_loss = loss(u(bound3_x_0, t), torch.zeros_like(t)) #u(-1, t) = 0

        #Add all the losses and return their values 
        return data_loss + phy_loss + bound1_loss + bound2_loss + bound3_loss
    
    return diffusion_loss