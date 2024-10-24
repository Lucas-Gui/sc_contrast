'''
Classes to organize the data for contrastive and supervised learning.
Data is stored in a BaseData object, which should live in the main process.
BaseData can be either Data or SlicedData, the latter being a view of a subset of the data.
Then, BaseData are passed to Dataset objects, which are used to create DataLoader objects.
'''

from dataclasses import dataclass
from torch import Tensor
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, SubsetRandomSampler
import pandas as pd
import numpy as np
from typing import *

@dataclass
class DataSample:
    '''Contains a sample of data, to be returned by SlicedData.__getitem__'''
    x: Tensor
    y: Tensor
    cycle: Tensor|None
    variant: str # as returned by accessing Data.variants

class Data():
    '''
    A class that contains the data and metadata for a dataloader.
    Lives in the main process, will be passed to the queue to be made into the appropriate DataLoader
    by the worker processes. 
    Allows to share x without copying.
    '''
    # cats : pd.Categorical
    # cycle_cats : pd.Categorical
    x : Tensor
    cycle : Tensor
    variants : pd.Series

    def __init__(self, x, cycle, variants ) -> None:
        self.x = x
        self.cycle = cycle
        # self.cycle_cats = cycle_cats
        self._variants = variants

    @property # protect against modification
    def variants(self) -> pd.Series:
        return self._variants
    
    def __len__(self):
        return len(self._variants)

    @classmethod 
    def from_df(cls, df:pd.DataFrame, device='cpu') -> Self:
        '''
        Initialize data for a dataset from a counts dataframe.
        Convert variant and cycle to labels and store categories.
        Yield ((n examples), (n labels) pairs)
        '''
        # cats = df.variant.cat.categories.copy() # seen variants are first
        # cycle_cats = df['cycle'].cat.categories.copy()
        cycle = torch.tensor(
            df['cycle'].cat.codes.to_numpy(), dtype=int, device=device
        )
        x = torch.tensor(
                df.drop(columns=['variant','Variant functional class', 'cycle']).to_numpy(), 
            dtype=torch.float32, device=device)
        # for instance bag datasets # TODO : make sure this is not an issue, otherwise move in child class
        variants = df.variant.copy() # to be able to group by
        data = cls(x, cycle, variants)
        return data



class SlicedData(Dataset):
    '''
    Map to a subset of a Data to allow models to train over different subsets of the data without copying.
    categorical variables are reordered to match the new subset.
    attributes:
        data : Data
        index : pd.DataFrame indexed by cell ids, sorted in the order of data slice, 
            such that index.j.iloc is a mapping from the new index to the original index.
            Contains columns 'j', 'i', and 'variant'. j should not be used outside of this class. 
            i can be used to query
    '''
    def __init__(self, data:Data, index:pd.Series|None, cats = None) -> None:
        '''
        args:
            data : Data
            index: boolean mask indexed by cell ids.
            cats : categories to use for the new data. If None, will use the categories of the original data.
        '''
        df = pd.DataFrame(data.variants)
        if index is None:
            index = pd.Series(True, index = df.index)
        df['j'] = np.arange(len(df)) # original order
        df = df[index]
        df['i'] = np.arange(len(df)) # useful for querying data with a specific variant
        if cats is not None:
            df['variant'] = df['variant'].cat.reorder_categories(cats, )

        self.index = df
        # self.index is in the same order as input index
        self.data = data    
        self.y = torch.tensor(
            df['variant'].cat.codes.to_numpy(), dtype=int, device=data.x.device
        )
    
    @property
    def variants(self)-> pd.Series:
        return self.index['variant']

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i) -> DataSample:
        j = self.index.j.iloc[i]
        return DataSample(
            self.data.x[j],
            self.y[i], # y is not in Data
            self.data.cycle[j],
            self.data.variants.iloc[j], # = self.index['variant'].iloc[i]
        )
    
    def get_x(self):
        '''Return a copy of the data X, in the order of the slice'''
        return self.data.x[self.index.j]
    

    

class _DfDataset(Dataset):
    def __init__(self, data:SlicedData, **kwargs) -> None:
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]: # return a pair (*x, *y) of tensor n-uplets, with the same n
        raise NotImplementedError


class InstanceBagDataset(IterableDataset, _DfDataset):
    '''
    Load data for Multiple Instance Learning.
    Yields (h,k) X tensors with h instances of the same class.
    '''
    def __init__(self, data:Data, bag_size=5, p=0.5, **kwargs) -> None:
        raise NotImplementedError('Need to be adapted to new SlicedData')
        _DfDataset.__init__(self, data) #this is ugly design but I don't want to rely on Pytorch being cooperative
        # cell ids is a dataframe with the same index as variants, which is a Series
        self.cell_ids = pd.DataFrame(np.arange(len(data)), index = data.variants.index.copy(), columns=['ids']) # numerical index w.r.t. X
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
    Yields (x1, x2) , (y,) tensors, where x1 and x2 have the same label y.
    Requires that all classes have at least two examples.
    """

    def __init__(self, data:Data, p=0.5, **kwargs) -> None:
        '''
        p is ignored
        '''
        super().__init__(data)
    
    def __getitem__(self, i1) -> Any:
        sample = self.data[i1]
        x1 = sample.x
        y = sample.y
        v = sample.variant
        idx = self.data.index
        y_subset = idx[(idx.variant == v)&(idx.i !=i1)].i
        i2 = torch.randint(0, len(y_subset), (1,)).item() # using random from torch 
        i2 = y_subset.iloc[i2]
        x2 = self.data[i2].x
        return (x1,x2), (y,) 
    
class SiameseDataset(_DfDataset):

    def __init__(self,  data:Data, p=0.5, **kwargs) -> None:
        '''
        p : probability to choose a positive pair at each pair sampling
        '''
        super().__init__(data)
        self.p = p
    
    def __getitem__(self, i1) -> Any:
        sample = self.data[i1]
        x1 = sample.x
        y1 = sample.y
        v = sample.variant
        if torch.rand((1,)).item() < self.p: # randomly get same class or different class
            # same class case : positive pair
            y_subset = self.data.index[ self.data.index.variant == v].i
        else:
            y_subset = self.data.index[(self.data.index.variant != v)&(self.data.index.i != i1)].i
        i2 = torch.randint(0, len(y_subset), (1,)).item()
        i2 = y_subset.iloc[i2]
        x2 = self.data[i2].x
        y2 = self.data[i2].y
        return (x1,x2), (y1, y2) #,i
    
class ClassifierDataset(_DfDataset):
    '''
    p is ignored
    '''
    def __init__(self, data:Data, p=None, **kwargs) -> None:
        super().__init__(data)

    def __getitem__(self, index) -> Any:
        sample = self.data[index]
        return (sample.x,), (sample.y,)
    

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

def make_loaders(*data:SlicedData, batch_size=64, dataset_class = SiameseDataset,  pos_frac=0.5,
                 subsampling_t = None,
                 dataset_kwargs={},
                 n_workers = 8, device='cpu', verbosity=2) -> Tuple[DataLoader]:
    '''
    pos_frac : fraction of positive pairs in training set (first dataloader.).
    Other dataloaders will have a 50% fraction of positive pairs.
    dataset_class : class for train dataset. Test datasets will always be SiameseDataset 
    subsampling_t: subsampling parameter for SubsamplingSampler for the training set. 
        If an integer, will be the maximum number of examples per variant.
        If a float, will be the quantile of the distribution of variant sizes to use as threshold.
    '''
    dls = []
    for i, d in enumerate(data):
        if d is None:
            dls.append(None)
            continue
    
        ds = dataset_class(d, p=pos_frac  if i ==0 else 0.5, **dataset_kwargs) # use half/half +/- pairs for eval (only for relevant dataloaders)
        if i == 0 and subsampling_t is not None:
            shuffle = False
            if isinstance(subsampling_t, int):
                sampler = SubsamplingSampler(d, max_size=subsampling_t)
            elif isinstance(subsampling_t, float):
                sampler = SubsamplingSampler(d, quantile=subsampling_t)   
            else:
                raise ValueError('subsampling_t should be an int or a float')
            if verbosity > 0:
                print('Subsampling training set to ',sampler.threshold, ' cells per variant')
        else:
            sampler = None
            if issubclass(dataset_class, InstanceBagDataset):
                shuffle = False 
                assert subsampling_t is None, 'Subsampling not implemented for InstanceBagDataset'
            else:
                shuffle = True    
        dls.append(DataLoader(ds, batch_size=batch_size, 
                              shuffle=shuffle, sampler=sampler,
                               num_workers=n_workers))
    return dls


class SubsamplingSampler(SubsetRandomSampler):
    r'''
    A sampler that subsamples variants that are too numerous in the dataset. 
    Variants above a given threshold are subsampled to that threshold.
    Args:
        data (Data): the data to sample from, that will be contained in the DataLoader.
        max_size (int): number of examples to use as threshold.
        quantile (float): the quantile of the distribution of variant sizes to use as threshold.
            Mutually exclusive with max_size.
    '''
    indices : pd.Index
    def __init__(self, data:SlicedData, max_size:int=None, quantile:float=None, generator=None ) -> None:
        if max_size is not None and quantile is not None:
            raise ValueError('max_size and quantile are mutually exclusive')
        self.variants = data.index[['variant','i']].copy()
        if quantile is not None:
            assert 0 < quantile < 1, 'quantile should be between 0 and 1'
            # unused categories add unwanted zeros
            self.threshold = int(self.variants['variant'].cat.remove_unused_categories().value_counts().quantile(quantile))
        else:
            self.threshold = max_size
        self.subsample() # initialize indices. Usually not needed except for length. 
        self.generator = generator

    def subsample(self) -> None:
        '''
        Compute the indices to sample from the dataset and store them in self.indices.
        '''
        larger = self.variants.groupby('variant', observed=True).filter(lambda x: len(x) > self.threshold) #observed to ignore empty categories
        larger = larger.groupby('variant', observed=True).sample(n=self.threshold, replace=False)
        smaller = self.variants.groupby('variant', observed=True).filter(lambda x: len(x) <= self.threshold)
        self.indices = pd.concat([larger.i, smaller.i], axis=0).values

    def __iter__(self):
        self.subsample()
        return super().__iter__()

