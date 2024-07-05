from torch import Tensor
import torch
import pandas as pd
from sklearn.model_selection import KFold
from typing import *


class MappedTensor():
    '''
    Implement a tensor as mapping of a parent tensor, without copying memory.
    '''

    def __init__(self, tensor:Tensor, mappings:List[Tensor|None]):
        '''
        tensor : parent tensor
        mappings : mapping tensors for each dimension. 
            Tensors of length tensor.shape[i] containing indices in the parent tensor
        '''
        # check dimension 
        assert len(mappings) == tensor.ndim
        if not all(m is None or (m.min() >=0 and m.max()<s) for m,s in zip(mappings, tensor.shape)):
            raise ValueError('Mappings must be None or contain valid indices')
        self.tensor = tensor
        self.mappings = mappings

    @property
    def shape(self):
        return tuple(len(m) if m is not None else s for m,s in zip(self.mappings, self.tensor.shape))
    
    def __getitem__(self, key):
        '''
        Key part of the new class.
        If keys are ints or tensors, return the corresponding value in the parent tensor.
        If keys are slices, raise NotImplementedError. 
        '''# TODO ? implement slices and ellipsis (could return a new MappedTensor with the same parent tensor, or a copy)
        if  isinstance(key, Tensor) or not isinstance(key, Iterable):
            key = (key,)
        new_key = []
        for i,k in enumerate(key):
            m = self.mappings[i]
            if m is None:
                new_key.append(k)
                continue
            if isinstance(k, int):
                new_key.append(m[k].item())
            elif isinstance(k, Tensor):
                new_key.append(m[k])
            elif isinstance(k, slice):
                raise NotImplementedError('Slices are not implemented')
            # elif isinstance(k, type(Ellipsis)): 
            else:
                raise ValueError(f'Unsupported key type {type(k)}')
        return self.tensor[tuple(new_key)]