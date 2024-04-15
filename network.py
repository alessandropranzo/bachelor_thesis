import torch
from torch import nn

import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

class LinearNN(nn.Module):
    
    def __init__(self, layers: list = [1, 5, 5, 5, 5, 1], act: nn.Module = nn.ReLU):
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

        layer_list.append("layer_%d" % (self.depth - 1), nn.Linear(layers[-2], layers[-1]))

        layer_dict = OrderedDict(layer_list)

        self.layers = nn.Sequential(layer_dict)

    def forward(self, x):
        out = self.layers(x)
        return x