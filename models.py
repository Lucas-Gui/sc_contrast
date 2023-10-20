"""Siamese, triplet and other contrastive network models"""

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn as nn
from typing import *


class Siamese(Module):
    '''
    A siamese network.
    args :
        network : inner module of the network. Must be correctly initialised. 
    '''
    def __init__(self, network) -> None:
        super().__init__()
        self.network = network

    def forward(self, x1: Tensor, x2:Tensor, ):
        """
        args:
            x1, x2 : Input tensors
        """
        e1 = self.network(x1)
        e2 = self.network(x2)
        return e1, e2

class SiameseLoss():
    '''
    args:
        alpha: l2 regularization coefficient applied to embeddings
    '''
    def __init__(self, alpha=1e-3) -> None:
        self.alpha = alpha

    def forward(self, e1: Tensor, e2: Tensor, y1, y2):
        """
        Compute the siamese loss.
        args:
            e1, e2 (Tensors): embeddings
            y1, y2 (any) : labels    
        """
        loss = 0
        loss = loss + self.alpha * (torch.norm(e1, dim=-1) + torch.norm(e2, dim=-1))
        raise NotImplementedError

class MLP(Module):
    def __init__(self, input_shape:int, inner_shape:Sequence[int]= (100,100), output_shape:int=20, act = nn.GELU()) -> None:
        super().__init__()
        shape = [input_shape, *inner_shape, ]
        modules = []
        for i in range(len(shape)-1):
            modules.append(nn.Linear(shape[i], shape[i+1]))
            modules.append(act)
        modules.append(nn.Linear(inner_shape[-1], output_shape))
        self.layers = nn.Sequential(*modules)

    def forward(self, x:Tensor) -> Tensor:
        return self.layers.forward(x)



