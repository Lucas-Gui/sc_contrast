from torch import Tensor
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pandas as pd
import numpy as np
from typing import *

class Data():
    '''
    A class that contains the data and metadata for a dataloader.
    Lives in the main process, will be passed to the queue to be made into the appropriate DataLoader
    by the worker processes. 
    '''
    cats : pd.Categorical
    cycle_cats : pd.Categorical
    x : Tensor
    y : Tensor
    cycle : Tensor
    cell_ids : pd.DataFrame
    variants : pd.Series

    def __init__(self, x,y,cycle, cats,cycle_cats, cell_ids,variants ) -> None:
        self.x =x
        self.y = y
        self.cycle = cycle
        self._cats = cats
        self.cycle_cats = cycle_cats
        self.cell_ids = cell_ids
        self._variants = variants
    # safety against changing categories after y has been computed
    @property
    def cats(self):
        return self._cats
    @cats.setter
    def cats(self, value):
        if self.y is not None:
            raise ValueError('Cannot change categories if y is not None')
        self._cats = value

    @property
    def variants(self):
        return self._variants
    @variants.setter
    def variants(self, value):
        if self.y is not None:
            raise ValueError('Cannot change categories if y is not None')
        self._variants = value
    
    def __len__(self):
        return len(self._variants)

    @classmethod 
    def from_df(cls, df:pd.DataFrame, device='cpu') -> Self:
        '''
        Initialize data for a dataset from a counts dataframe.
        Convert variant and cycle to labels and store categories.
        Yield ((n examples), (n labels) pairs)
        '''
        cats = df.variant.cat.categories.copy() # seen variants are first
        cycle_cats = df['cycle'].cat.categories.copy()
        cycle = torch.tensor(
            df['cycle'].cat.codes.to_numpy(), dtype=int, device=device
        )
        x = torch.tensor(
                df.drop(columns=['variant','Variant functional class', 'cycle']).to_numpy(), 
            dtype=torch.float32, device=device)
        y = None
        # for instance bag datasets # TODO : make sure this is not an issue, otherwise move in child class
        # cell_ids : bc df indexes are cell barcodes
        cell_ids = pd.DataFrame(np.arange(len(df)), index = df.index.copy(), columns=['ids'])  
        variants = df.variant.copy() # to be able to group by
        data = cls(x,y,cycle, cats,cycle_cats, cell_ids, variants)
        return data
    
    def subset(self, index, drop_cats = False) -> Self:
        '''
        Create a new Data instance referring to a subset of the data.
        index should be a boolean mask (either an array or a series indexed by cell ids).
        If drop_cats, unused categories will be dropped from the categories.
        '''
        data = Data(
            self.x[index],
            self.y[index] if self.y is not None else None,
            self.cycle[index],
            self.cats,
            self.cycle_cats,
            self.cell_ids.loc[index],
            self.variants.loc[index]
        )
        if drop_cats:
            if self.y is None:
                data.variants = data.variants.cat.remove_unused_categories()
                data.cats = data.variants.cat.categories
            else:
                raise ValueError('Cannot drop categories if y is not None')
        return data
    
    def compute_y(self):
        '''
        Compute y tensor from the variant categories.
        Separated from constructor to allow category reordering and subsampling.
        '''
        device = self.x.device
        self.y = torch.tensor(
            self.variants.cat.codes.to_numpy(), dtype=int, device=device
        )

    # def subsample_variants(self, n) -> Self:
    #     '''
    #     Filter Data tensors in-place to keep only n variants.
    #     '''
    #     variants = self.variants.cat.categories
    #     variants = np.random.choice(variants,replace=False, size = n )
    #     filt = self.variants.isin(variants)
    #     self.x = self.x[filt]
    #     self.y = self.y[filt]
    #     self.cycle = self.cycle[filt]
    #     self.cell_ids = self.cell_ids[filt]


class _DfDataset(Dataset):
    def __init__(self, data:Data, **kwargs) -> None:
        self.x = data.x
        self.y = data.y
        self.cycle = data.cycle
        self.cats = data.cats
        self.cycle_cats = data.cycle_cats
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]: # return a pair (*x, *y) of tensor n-uplets, with the same n
        raise NotImplementedError


class InstanceBagDataset(IterableDataset, _DfDataset):
    '''
    Load data for Multiple Instance Learning.
    Yields (h,k) X tensors with h instances of the same class.
    '''
    def __init__(self, data:Data, bag_size=5, p=0.5, **kwargs) -> None:
        _DfDataset.__init__(self, data) #this is ugly design but I don't want to rely on Pytorch being cooperative
        # cell ids is a dataframe with the same index as variants, which is a Series
        self.cell_ids = data.cell_ids 
        self.variant = data.variants
        self.bag_size = bag_size

    def make_bags(self):
        '''Randomly group the examples into same-label bags
        Return Tensors X and y with shape (batch_size, bag_size, n_features) and (batch_size,) respectively
        and a dataframe with the cell ids for each bag
        ''' 
        # First shuffle so that bags are different each epoch
        shuffled = self.cell_ids.sample(frac=1)
        # then group by variant and yield bags
        groups = shuffled.groupby(self.variant)
        dfs = []
        for name, group in groups:
            group = group.head(self.bag_size * (len(group) // self.bag_size)) # drop last bag if not full
            dfs.append(group)
        bags = pd.concat(dfs)
        X = self.x[bags['ids']]
        y = self.y[bags['ids']]
        X = X.reshape(-1, self.bag_size, X.size(1))
        y = y[::self.bag_size]
        # make cell id table (batch nb, instance nb)
        bags['instance'] = np.tile(np.arange(self.bag_size), len(bags) // self.bag_size)
        bags['batch'] = np.repeat(np.arange(len(bags) // self.bag_size), self.bag_size)
        bags = bags.pivot(index='batch', columns='instance', values='ids')
        print(bags.shape)
        # shuffle bags
        idx = torch.randperm(len(y))
        X = X[idx]
        y = y[idx]
        bags = bags.iloc[idx]
        return X, y, bags
    
    def __iter__(self):
        return iter(self.generate())
    
    def generate(self):
        raise NotImplementedError
    
    def __len__(self):
        return self.cell_ids.groupby(self.variant).size().floordiv(self.bag_size).sum() 

class ClassifierBagDataset(InstanceBagDataset):
    '''
    Load data for Multiple Instance Learning classification.
    '''
    def generate(self):
        X,y, _ = self.make_bags()
        for x,y_ in zip(X,y):  
            yield (x,), (y_,)  

class ClassifierBagDatasetWithIDs(InstanceBagDataset): #not sure if making two classes is useful. 
    '''
    Load data for Multiple Instance Learning classification, with cell ids for analysis.
    '''
    def generate(self):
        X,y, cell_ids = self.make_bags()
        for x,y_, cell in zip(X,y, cell_ids.values):  
            yield (x,), (y_, cell)  

class BatchBagDataset(InstanceBagDataset):
    '''
    Load data for Multiple Instance Learning batch contrastive learning.
    '''
    def generate(self):
        X,y, _ = self.make_bags()
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

    def __init__(self, data:Data, p=0.5, **kwargs) -> None:
        '''
        p is ignored
        '''
        super().__init__(data)
    
    def __getitem__(self, index) -> Any:
        x1 = self.x[index]
        y = self.y[index]
        y_subset = self.y[self.y == y]
        i = torch.randint(0, y_subset.size(0), (1,)).item()
        x2 = self.x[i]
        return (x1,x2), (y,) 
    
class SiameseDataset(_DfDataset):

    def __init__(self,  data:Data, p=0.5, **kwargs) -> None:
        '''
        p : probability to choose a positive pair at each pair sampling
        '''
        super().__init__(data)
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
    def __init__(self, data:Data, p=None, **kwargs) -> None:
        super().__init__(data)

    def __getitem__(self, index) -> Any:
        return (self.x[index],), (self.y[index],) 
    

class BipartiteDataset(Dataset): #TODO : conform to superclass return scheme for compatibility
    '''
    A dataset that takes two dataframes as data.
    Iterates over df1, and samples a second random example, either from df1, in the same class as x1 (with probability p1) or df2
    '''
    # Positive examples are still same-label pairs and NOT pairs with both examples from df1.
    def __init__(self, df1:pd.DataFrame, df2:pd.DataFrame, p1=0.5, device='cpu') -> None:
        raise NotImplementedError('This class is deprecated')
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
    def __getitem__(self, index) -> Any:
        return (self.x[index],), (self.y[index], self.cycle[index]) 

def make_loaders(*dfs:pd.DataFrame, batch_size=64, dataset_class = SiameseDataset,  pos_frac=0.5,
                 dataset_kwargs={},
                 n_workers = 8, device='cpu') -> Tuple[DataLoader]:
    '''
    pos_frac : fraction of positive pairs in training set (first dataloader.).
    Other dataloaders will have a 50% fraction of positive pairs.
    dataset_class : class for train dataset. Test datasets will always be SiameseDataset  
    '''
    dls = []
    for i, df in enumerate(dfs):
        if df is None:
            dls.append(None)
            continue
        ds = dataset_class(df, p=pos_frac  if i ==0 else 0.5, **dataset_kwargs) # use half/half +/- pairs for eval (only for relevant dataloaders)
        dls.append(DataLoader(ds, batch_size=batch_size, shuffle=True if not issubclass(dataset_class, InstanceBagDataset) else None,
                               num_workers=n_workers))
    return dls




