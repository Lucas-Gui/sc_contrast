from torch import Tensor
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pandas as pd
import numpy as np
from typing import *

#TODO : do this with multiple inheritance
def init_dataset_from_dataframe(self, df:pd.DataFrame, device):
    '''
    Initialize a dataset from a dataframe.
    Convert variant and cycle to labels and store categories.
    '''
    variants = df['variant']
    self.cats = variants.cat.categories
    self.y = torch.tensor(
        variants.cat.codes.to_numpy(), dtype=int, device=device
    )
    self.cycle_cats = df['cycle'].cat.categories
    self.cycle = torch.tensor(
        df['cycle'].cat.codes.to_numpy(), dtype=int, device=device
    )
    self.x = torch.tensor(
            df.drop(columns=['variant','Variant functional class', 'cycle']).to_numpy(), 
        dtype=torch.float32, device=device)
    
class _DfDataset(Dataset):
    def __init__(self, df:pd.DataFrame, device) -> None:
        '''
        Yield ((n examples), (n labels) pairs)'''
        super().__init__()
        init_dataset_from_dataframe(self, df, device)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]: # return a pair (*x, *y) of tensor n-uplets, with the same n
        raise NotImplementedError


class InstanceBagDataset(IterableDataset):
    '''
    Load data for Multiple Instance Learning.
    Yields (h,k) X tensors with h instances of the same class.
    '''
    def __init__(self, df:pd.DataFrame, device='cpu', bag_size=5) -> None:
        super().__init__()
        init_dataset_from_dataframe(self, df, device)
        self.df = df
        self.bag_size = bag_size
        self.device = device

    def make_bags(self):
        '''Randomly group the examples into same-label bags
        Return a list of DataFrames (bags) and a DataFrame mapping bag index to bag label''' 
        # First shuffle so that bags are different each epoch
        shuffled = self.df.sample(frac=1)
        # then group by variant and yield bags
        groups = shuffled.groupby('variant')
        dfs = []
        for name, group in groups:
            group = group.head(self.bag_size * (len(group) // self.bag_size)) # drop last bag if not full
            dfs.append(group)
        bags = pd.concat(dfs)
        X = torch.tensor(
            bags.drop(columns=['variant','Variant functional class', 'cycle']).to_numpy(), 
        dtype=torch.float32, device=self.device)
        y = torch.tensor(
            bags.variant.cat.codes.to_numpy(), dtype=int, device=self.device
        )
        X = X.reshape(-1, self.bag_size, X.size(1))
        y = y[::self.bag_size]
        # shuffle bags
        idx = torch.randperm(len(y))
        X = X[idx]
        y = y[idx]
        return X, y
    
    def __iter__(self):
        return iter(self.generate())
    
    def generate(self):
        raise NotImplementedError
    
    def __len__(self):
        return self.df.groupby('variant').size().floordiv(self.bag_size).sum() 

class ClassifierBagDataset(InstanceBagDataset):
    '''
    Load data for Multiple Instance Learning classification.
    '''
    def generate(self):
        X,y = self.make_bags()
        return zip(X, y)

class BatchBagDataset(InstanceBagDataset):
    '''
    Load data for Multiple Instance Learning batch contrastive learning.
    '''
    def __init__(self, df: pd.DataFrame, p=0.5, device='cpu', bag_size=5) -> None:
        '''
        p : probability to choose a positive pair at each pair sampling
        '''
        super().__init__(df, device=device, bag_size=bag_size)
        self.p = p

    def generate(self):
        X,y = self.make_bags()
        for i in range(len(y)):
            x1 = X[i]
            y1 = y[i]
            y_subset = y[y == y1]
            i = torch.randint(0, y_subset.size(0), (1,)).item()
            x2 = X[i]
            yield (x1,x2), (y1,) 
    

class BatchDataset(_DfDataset):
    """
    Load data for batch contrastive loss (Khosla et al, )
    Because we don't do data augmentation, instead we randomly sample pairs of cells from the same classes.
    Yields (x1, x2) , (y,) tensors, where x1 and x2 have the same label y
    """

    def __init__(self, df: pd.DataFrame, p=0.5, device='cpu') -> None:
        '''
        p : probability to choose a positive pair at each pair sampling
        '''
        super().__init__(df, device=device)
        self.p = p
    
    def __getitem__(self, index) -> Any:
        x1 = self.x[index]
        y = self.y[index]
        y_subset = self.y[self.y == y]
        i = torch.randint(0, y_subset.size(0), (1,)).item()
        x2 = self.x[i]
        return (x1,x2), (y,) 
    
class SiameseDataset(_DfDataset):

    def __init__(self, df: pd.DataFrame, p=0.5, device='cpu') -> None:
        '''
        p : probability to choose a positive pair at each pair sampling
        '''
        super().__init__(df, device=device)
        self.p = p
    
    def __getitem__(self, index) -> Any:
        x1 = self.x[index]
        y1 = self.y[index]
        if torch.rand((1,)).item() < self.p: # randomly get same class or different class
            # same class case : positive pair
            y_subset = self.y[self.y == y1]
        else:
            y_subset = self.y[self.y != y1]
        i = torch.randint(0, y_subset.size(0), (1,)).item()
        x2 = self.x[i]
        y2 = self.y[i]
        return (x1,x2), (y1, y2) #,i
    
class ClassifierDataset(_DfDataset):
    '''
    p is ignored
    '''
    def __init__(self, df: pd.DataFrame, p=None, device='cpu') -> None:
        super().__init__(df, device=device)

    def __getitem__(self, index) -> Any:
        return (self.x[index],), (self.y[index],) 
    

class BipartiteDataset(Dataset): #TODO : conform to superclass return scheme for compatibility
    '''
    A dataset that takes two dataframes as data.
    Iterates over df1, and samples a second random example, either from df1, in the same class as x1 (with probability p1) or df2
    '''
    # Positive examples are still same-label pairs and NOT pairs with both examples from df1.
    def __init__(self, df1:pd.DataFrame, df2:pd.DataFrame, p1=0.5, device='cpu') -> None:
        super().__init__()
        self.y1 = df1['variant'].reset_index(drop=True)
        self.x1 = torch.tensor(
                df1.drop(columns=['variant','Variant functional class']).to_numpy(), 
            dtype=torch.float32, device=device)
        self.y2 = df2['variant'].reset_index(drop=True)
        self.x2 = torch.tensor(
                df2.drop(columns=['variant','Variant functional class']).to_numpy(), 
            dtype=torch.float32, device=device)
        self.p1 = p1
    
    def __len__(self):
        return len(self.y1)
    
    def __getitem__(self, index) -> Any:
        x1 = self.x1[index]
        y1 = self.y1.iloc[index]
        if torch.rand((1,)).item() < self.p1: # randomly get same class or different class
            # same class case : positive pair
            y2 = self.y1[self.y1 == y1].sample(1)
            x2 = self.x1[y2.index.item()]
        else:
            y2 = self.y2.sample(1)
            x2 = self.x2[y2.index.item()]

        y = torch.tensor(y1==y2.item())
        return (x1,x2), y #,i
    
class CycleClassifierDataset(_DfDataset):
    '''Return two labels, variant and cell cycle'''
    def __init__(self, df: pd.DataFrame, p=None, device='cpu') -> None:
        super().__init__(df, device=device)

    def __getitem__(self, index) -> Any:
        return (self.x[index],), (self.y[index], self.cycle[index]) 

def make_loaders(*dfs:pd.DataFrame, batch_size=64, dataset_class = SiameseDataset, n_workers = 8, pos_frac=0.5, device='cpu') -> Tuple[DataLoader]:
    '''
    pos_frac : fraction of positive pairs in training set (first dataloader.).
    Other dataloaders will have a 50% fraction of positive pairs.
    dataset_class : class for train dataset. Test datasets will always be SiameseDataset  
    '''
    dls = []
    for i, df in enumerate(dfs):
        if len(df) == 0:
            dls.append(None)
            continue
        ds = dataset_class(df, p=pos_frac  if i ==0 else 0.5) # use half/half +/- pairs for eval (only for relevant dataloaders)
        dls.append(DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=n_workers))
    return dls
