from torch import Tensor
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import *

class _DfDataset(Dataset):
    def __init__(self, df:pd.DataFrame) -> None:
        super().__init__()
        self.y = df['variant']
        self.x = df.drop(columns=['variant','Variant functional class'])
    
    def __len__(self):
        return len(self.y)
    
    # def __getitem__(self, index) -> Tuple[Tuple[Tensor]]:
    #     raise NotImplementedError
    
class SiameseDataset(_DfDataset):
    
    def __getitem__(self, index) -> Any:
        x1 = torch.tensor(self.x.iloc[index].to_numpy(), dtype=torch.float32)
        y1 = self.y.iloc[index]
        if torch.rand((1,)).item() > 0.5: # randomly get same class or different class
            # same class case
            x2 = self.x[self.y == y1].sample(1)
            y2 = y1
        else:
            x2 = self.x[self.y != y1].sample(1)
            y2 = self.y.loc[x2.index].item()
        x2 = torch.tensor(x2.to_numpy(), dtype=torch.float32)
        y = torch.tensor(y1==y2)
        return (x1,x2), y

def make_loaders(*dfs:pd.DataFrame, batch_size=64, dataset_class = SiameseDataset, n_workers = 4):
    dls = []
    for df in dfs:
        ds = dataset_class(df)
        dls.append(DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=n_workers))
    return dls