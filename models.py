"""Siamese, triplet and other contrastive network models"""

from typing import Any
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as f
from typing import *
import pandas as pd

# Loss

class ContrastiveLoss():
    alpha:float
    margin:float

    def __init__(self, margin, alpha) -> None:
        self.alpha = alpha
        self.margin = margin
    
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
    def forward(self, e1: Tensor, e2: Tensor, y1 : Tensor, y2 : Tensor):
        """
        Compute the siamese loss.
        args:
            e1, e2 (Tensors): embeddings
            y1, y2 (any) : labels    
        """
        y = y1 == y2
        loss = 0 # it would be prettier to have the embedding penalty somewhere else, but whatever
        loss = loss + self.alpha * (torch.norm(e1, dim=-1) + torch.norm(e2, dim=-1))/2
        d = torch.norm((e1-e2), p=2, dim=-1)
        loss = loss + y*d**2 + (~y)*torch.maximum(self.margin - d, torch.zeros_like(d))**2
        return loss.mean()
    
class ClassifierLoss(ContrastiveLoss):
    '''
    CrossEntropy wrapper
    '''
    def forward(self, ypred, y) -> Tensor:
        return f.cross_entropy(ypred, y, reduction='mean')
    

class DoubleClassifierLoss(ContrastiveLoss):
    '''
    Cross entropy on double classification task
    '''
    def __init__(self, margin, alpha) -> None:
        super().__init__(margin, alpha)
        assert 0<= self.alpha<=1

    def __repr__(self) -> str:
        return super().__repr__()+ f"\nalpha : {self.alpha}" 

    def forward(self, ypred_1, ypred_2, y1, y2) -> Tensor:
        l1 =  f.cross_entropy(ypred_1, y1, reduction='mean')
        l2 =  f.cross_entropy(ypred_2, y2, reduction='mean')
        return self.alpha *l1 + (1-self.alpha)*l2
    
class LeCunContrastiveLoss(ContrastiveLoss):
    '''
    $L = (Y)\\frac{2}{Q} ||e_1 - e_2||_1^2 + (1-Y) 2Q\exp{-\\frac{2.77}{Q}||e_1 - e_2||_1}$
    '''

    def forward(self, e1: Tensor, e2: Tensor, y1 : Tensor,  y2 : Tensor):
        """
        Compute the siamese loss.
        args:
            e1, e2 (Tensors): embeddings
            y1, y2 (any) : labels    
        """
        y = y1 == y2
        loss = 0
        loss = loss + self.alpha * (torch.norm(e1, dim=-1) + torch.norm(e2, dim=-1))/2
        d = torch.norm((e1-e2), p=1, dim=-1)
        loss = loss + y* d**2 *2/self.margin + (~y)* 2*self.margin*torch.exp(-2.77/self.margin*d)
        return loss.mean()
    
class CosineContrastiveLoss(ContrastiveLoss):
    '''
    $L = Y\\ \\d(e_1, e_2) + (1-Y) (1 - d(e_1, e_2))$
    '''

    def forward(self, e1: Tensor, e2: Tensor, y1 : Tensor,  y2 : Tensor):
        """
        Compute the siamese loss.
        args:
            e1, e2 (Tensors): embeddings
            y1, y2 (any) : labels    
        """
        y = y1 == y2
        e1 = e1/e1.norm(p=2, dim=-1, keepdim=True)
        e2 = e2/e2.norm(p=2, dim=-1, keepdim=True)
        loss = 0
        loss = loss + self.alpha * (torch.norm(e1, dim=-1) + torch.norm(e2, dim=-1))/2
        d = 1 - (e1 * e2).sum(dim=-1)
        loss = loss + y* d + (~y)* (1-d)
        return loss.mean()
    

class BatchContrastiveLoss(ContrastiveLoss):
    '''
    See Khosla et al, 2020
    '''

    def forward(self, e1:Tensor, e2:Tensor, y:Tensor) -> Tensor:
        '''
        Input : e1, e2, embeddings of exemples with labels y
        '''
        y = torch.concat((y,y))
        embeddings = torch.concat((e1, e2))
        positive = y[None, :].eq(y[:, None]) # positive[i,j] := y_i == y_j

        Z = embeddings @ embeddings.T / self.margin 
        Z = Z.exp() # Z(i,j) = exp( z_i . z_j /\tau)

        A = (Z * (~ torch.eye(Z.shape[0], dtype=bool, device=Z.device)))
        A = A.sum(dim=-1, keepdim=True) # \sum_{a \in A(i)} exp(z_i z_a/t) # shape N,1
        
        P = ((Z / A).log() * positive).sum(-1) # \sum_{p \in P(i)} \frac{\exp(z_i z_p/t)}{\sum_{a \in A(i)} \exp(z_i z_a/t)}
        P = - P/ positive.sum(dim=-1)
        loss = P # original is sum but we want to divide by the number of examples
        loss = loss + self.alpha * torch.norm(embeddings, dim=-1)
        return loss.mean()
    

class Model(Module):
    '''
    A network that can produce embeddings
    '''
    def __init__(self, network : nn.Module, normalize=True, **kwargs) -> None: #captures **kwargs to pass paramters that are specific to some model types
        super().__init__()
        self.network = network
        self.normalize = normalize

    def embed(self, x:Tensor) -> Tensor:
        x = self.network(x)
        if self.normalize:
            x = x/torch.norm(x, keepdim=True, dim=-1)
        return x
    
    def forward(self, *x : List[Tensor]) -> Tuple[Tuple[Tensor, ...], Tensor]:
        raise NotImplementedError
    
class Siamese(Model):
    '''
    A siamese network.
    args :
        network : inner module of the network. Must be correctly initialised. 
    '''

    def forward(self, x1: Tensor, x2:Tensor, ):
        """
        args:
            x1, x2 : Input tensors
        """
        e1 = self.embed(x1)
        e2 = self.embed(x2)
        return (e1, e2), e1


class Classifier(Model):
    '''
    A model that learns embeddings through a classification task.
    Implements a logistic regression on top of the embedding
    args :
        network : inner module of the network. Must be correctly initialised. 
    '''
    def __init__(self, network, n_class, **kwargs) -> None:
        super().__init__(network, **kwargs)
        self.output_layer = nn.Linear(network.output_shape, n_class)

    def forward(self, x: Tensor, ):
        """
        args:
            x: Input tensor
        """
        x = self.embed(x)
        logits = self.output_layer(x)
        return (logits,), x
    

class CycleClassifier(Model):
    '''
    A model that learns embeddings through a double classification task : cell cycle and variant.
    args :
        network : inner module of the network. Must be correctly initialised. 
    '''
    def __init__(self, network, n_class, **kwargs) -> None:
        super().__init__(network, **kwargs)
        self.output_layer_1 = nn.Linear(network.output_shape, n_class)
        self.output_layer_2 = nn.Linear(network.output_shape, 6)

    def forward(self, x: Tensor, ):
        """
        args:
            x: Input tensor
        """
        x = self.embed(x)
        logits_1 = self.output_layer_1(f.relu(x))
        logits_2 = self.output_layer_2(f.relu(x))
        return (logits_1, logits_2), x

    
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
        self.output_shape=output_shape

    def forward(self, x:Tensor) -> Tensor:
        x = self.layers.forward(x)

        return x
