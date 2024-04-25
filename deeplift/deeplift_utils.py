'''Module to retrieve the data.'''
import pandas as pd
import numpy as np
import time
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sb

from scipy.stats import uniform
from sklearn.decomposition import PCA

## PLOTS
def confusion_matrix(Y_true : pd.DataFrame, Y_pred : pd.DataFrame, 
    title = 'Confusion matrix', savepath = None, size=None
        ):
    scores = pd.DataFrame(data=0, index=Y_pred.columns, columns=Y_true.columns)
    N = Y_true.shape[0]
    assert Y_pred.shape[0]==N
    Y_pred.index = Y_true.index

    s= Y_pred.astype(int).T @ Y_true.astype(int) #get_dummies yield uint8, which apparently doesn't work with matmul ????

    scores.iloc[:,:] = s/N

    sb.heatmap(data=scores, annot=True, cmap='crest_r')
    plt.xlabel('True allele')
    plt.ylabel('Predicted')
    plt.title(title)
    if size is not None:
        plt.gcf().set_size_inches(size)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()

    return scores

## DATA LOADING
class Timer():
    def __init__(self, name=''):
        self.name=name
        self.t0=0
    
    def __enter__(self):
        self.t0=time.time()
    
    def __exit__(self,exc_type, exc_val, exc_tb):
        print(time.time()-self.t0, self.name, flush=True)
        return exc_type is None

def remove_cst(df : pd.DataFrame, thresh = 5e-3) -> pd.DataFrame:
    '''Removes columns that are always, or more often than thresh, zero'''
    count = pd.Series(index=df.columns, data=0)
    for _, sample in df.iterrows():
        count[sample.abs() > 1e-3] +=1
    return df.loc[:,count > thresh * df.shape[0]]

def get_p53(dataset_path, labelled, normalize:Literal['standard','max','mean'], thresh):
    with Timer('Read csv'):
        df = pd.read_csv(dataset_path, index_col=0, dtype={'allele':str, 'Radiation':pd.Int64Dtype(), 'Drug':pd.Int64Dtype()})
        X = df.iloc[:,:-3]
    if thresh is not None:
        with Timer('remove_cst'):
            X = remove_cst(X, thresh)
    if normalize is not None:
        with Timer('normalize'):
            X = reduce(X, normalize)
    X['allele'] = df.allele
    X['Radiation'] = df.Radiation
    X['Drug'] = df.Drug 

    if labelled:
        X = X.dropna(axis=0)
    return X

def reduce(df, method='standard'):
    '''Method in "standard", "max", "mean"
    "max" should only be used for nonnegative input, which is the case of our expression data'''
    if method == 'standard':
        return (df-df.mean())/df.std()
    elif method == 'max':
        return df/df.max()
    elif method=='mean':
        return df/df.mean()
    else :
        raise ValueError(method + ' is not a valid reduction')




