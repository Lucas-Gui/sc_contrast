from torch import Tensor
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import *

class _DfDataset(Dataset):
    def __init__(self, df:pd.DataFrame) -> None:
        super().__init__()
        self.y = df['variant'].reset_index(drop=True)
        self.x = torch.tensor(
                df.drop(columns=['variant','Variant functional class']).to_numpy(), 
            dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    # def __getitem__(self, index) -> Tuple[Tuple[Tensor]]:
    #     raise NotImplementedError
    
class SiameseDataset(_DfDataset):
    
    def __getitem__(self, index) -> Any:
        x1 = self.x[index]
        y1 = self.y.iloc[index]
        if torch.rand((1,)).item() > 0.5: # randomly get same class or different class
            # same class case
            i = self.y[self.y == y1].sample(1).index.item()
            y = True
        else:
            i = self.y[self.y != y1].sample(1).index.item()
            y = False

        x2 = self.x[i]
        y = torch.tensor(y)
        return (x1,x2), y #,i

def make_loaders(*dfs:pd.DataFrame, batch_size=64, dataset_class = SiameseDataset, n_workers = 8):
    dls = []
    for df in dfs:
        if len(df) ==0:
            dls.append(None)
            continue
        ds = dataset_class(df)
        dls.append(DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=n_workers))
    return dls