from torch import Tensor
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
import pandas as pd
from typing import *

class _DfDataset(Dataset):
    def __init__(self, df:pd.DataFrame, device) -> None:
        '''
        Yield ((n examples), (n labels) pairs)'''
        super().__init__()
        variants = df['variant']
        self.cats = variants.cat.categories
        self.y = torch.tensor(
            variants.cat.codes.to_numpy(), dtype=int, device=device
        )
        self.x = torch.tensor(
                df.drop(columns=['variant','Variant functional class']).to_numpy(), 
            dtype=torch.float32, device=device)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]: # return a pair (*x, *y) of tensor n-uplets, with the same n
        raise NotImplementedError

class BatchClassDataset(_DfDataset):
    """
    Load"""
    
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
