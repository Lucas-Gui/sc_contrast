"""Siamese, triplet and other contrastive network models"""

from typing import Any
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn as nn
from typing import *

# Loss

class ContrastiveLoss():
    alpha:float
    margin:float
    def forward(self, embeddings, y) -> Tensor:
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwds: Any) -> Tensor:
        return self.forward(*args, **kwds)


class SiameseLoss(ContrastiveLoss):
    '''
    args:
        margin: distance threshold for different classes
        alpha: l2 regularization coefficient applied to embeddings
    '''
    def __init__(self, margin, alpha) -> None:
        self.alpha = alpha
        self.margin = margin

    def forward(self, e1: Tensor, e2: Tensor, y : Tensor):
        """
        Compute the siamese loss.
        args:
            e1, e2 (Tensors): embeddings
            y1, y2 (any) : labels    
        """
        loss = 0 # it would be prettier to have the embedding penalty somewhere else, but whatever
        loss = loss + self.alpha * (torch.norm(e1, dim=-1) + torch.norm(e2, dim=-1))/2
        d = torch.norm((e1-e2), p=2, dim=-1)
        loss = loss + y*d**2 + (~y)*torch.maximum(self.margin - d, torch.zeros_like(d))**2
        return loss.mean()
    
class LeCunContrastiveLoss():
    '''
    $L = (Y)\\frac{2}{Q} ||e_1 - e_2||_1^2 + (1-Y) 2Q\exp{-\\frac{2.77}{Q}||e_1 - e_2||_1}$
    '''
    def __init__(self, margin, alpha) -> None:
        self.alpha = alpha
        self.margin = margin

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def forward(self, e1: Tensor, e2: Tensor, y : Tensor):
        """
        Compute the siamese loss.
        args:
            e1, e2 (Tensors): embeddings
            y1, y2 (any) : labels    
        """
        loss = 0
        loss = loss + self.alpha * (torch.norm(e1, dim=-1) + torch.norm(e2, dim=-1))/2
        d = torch.norm((e1-e2), p=1, dim=-1)
        loss = loss + y* d**2 *2/self.margin + (~y)* 2*self.margin*torch.exp(-2.77/self.margin*d)
        return loss.mean()


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
    
class MLP(Module):
    def __init__(self, input_shape:int, inner_shape:Sequence[int]= (100,100), 
                 output_shape:int=20, act = nn.ELU(), dropout = None) -> None:
        super().__init__()
        shape = [input_shape, *inner_shape, ]
        modules = []
        for i in range(len(shape)-1):
            modules.append(nn.Linear(shape[i], shape[i+1]))
            modules.append(act)
            if dropout :
                modules.append(nn.Dropout(p = dropout))
        modules.append(nn.Linear(inner_shape[-1], output_shape))
        self.layers = nn.Sequential(*modules)

    def forward(self, x:Tensor) -> Tensor:
        return self.layers.forward(x)



