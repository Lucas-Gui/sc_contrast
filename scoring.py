from sklearn.neighbors import KNeighborsClassifier, KNeighborsTransformer, sort_graph_by_row_values
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from scipy.sparse import csr_matrix, find
from scipy.stats import mode
from torch import Tensor
import torch
from models import Model
import numpy as np
from torch.utils.data import Dataset, DataLoader


def accuracy(e1: Tensor, e2: Tensor, y, margin):
    d = (e1 - e2).norm(p=2)
    return torch.logical_xor(y, d>margin).float().mean()

def ROC_score(y:Tensor, d:Tensor, eps = 1e-2):
    '''
    args:
        d (Tensor): example pair distances
        y (Tensor): example pair labels (1 if example belongs to the same class)
    '''
    d = (d - d.min())/(d.max() - d.min())
    s = y.mean(dtype=torch.float32)
    t = torch.arange(0,1,eps).reshape((-1,1)) #classification thresholds (T, 1) -> will broadcast to (T, N) with d and y
    tpr = ((d<t)*y).mean(dim=1, dtype=torch.float32)/s # (d<t) : predicted true; y : actual true; s : sum of actual true
    fpr = ((d<t)* (~y)).mean(dim=1, dtype=torch.float32)/(1-s) # (d<t) : predicted true; y : actual false; 1-s : sum of actual true
    # integration by sum of trapezes
    roc = 1/2 * ((fpr[1:] - fpr[:-1])*(tpr[1:] + tpr[:-1])).sum() 
    return roc, tpr, fpr

def knn_ref_score(model:Model, x_train:DataLoader, x_test:DataLoader, k=3, device='cpu'):
    '''
    Compute subset accuracy of knn classification, using x_train embeddings as the reference and x_test as the example to classify
    NB : x_train will be stored in memory in tensor form to compute the knns
    '''
    y_test, y_pred, _ = knn_ref_predict(model, x_train, x_test, k, device)
    return accuracy_score(y_test, y_pred)

def knn_ref_predict(model:Model, train:DataLoader, test:DataLoader, k, device='cpu', return_x = False):
    '''
    Compute predictions of ref-knn classification, using x_train embeddings as the reference and x_test as the examples to classify.
    Use knn_ref_score to compute the accuracy, and this function to get the predictions, inputs, and embeddings
    '''
    model.eval()
    model = model.to(device)
    emb_train, y_train, X_train = [], [], []
    for x,y in train: # x,y are tuples of tensors of any size, we're taking the first one
        with torch.no_grad():
            emb_train.append(model.embed(x[0].to(device)).cpu())#.squeeze() # does it do anything ? if yes, add torch.at_least_2d for the rare case where the last batch has size 1
            y_train.append(y[0])
            if return_x:
                X_train.append(x[0])
    emb_test, y_test, X_test = [], [], []
    for x,y in test:
        # print(x[0].shape, y[0].shape, flush=True)
        with torch.no_grad():
            emb_test.append(model.embed(x[0].to(device)).cpu())#.squeeze() # see above
            y_test.append(y[0])
            if return_x:
                X_test.append(x[0])
    emb_train = torch.concat(emb_train)
    emb_test = torch.concat(emb_test)
    y_train = torch.concat(y_train)
    y_test = torch.concat(y_test).cpu().numpy()
    if return_x:
        X_train = torch.concat(X_train)
        X_test = torch.concat(X_test)
    knn = KNeighborsClassifier(k)
    knn.fit(emb_train.cpu().numpy(), y_train.cpu().numpy())
    y_pred = knn.predict(emb_test.cpu().numpy())
    return y_test, y_pred, (y_train, X_train, X_test, emb_train, emb_test)

@torch.no_grad
def knn_self_score(embed:Tensor, y:Tensor,k=3):
    '''
    Compute subset accuracy of knn classification, using x_j (j != i) to classify each x_i
    '''
    knn = KNeighborsTransformer(n_neighbors=k)
    y = y.cpu().numpy()
    knn.fit(embed.cpu(), )
    graph = knn.kneighbors_graph()
    y_pred = y[find(graph)[1]] # neighbors' y (find(graph)[1] are the target nodes in the knn graph)
    y_pred = y_pred.reshape((-1, k)) # (kN, ) -> (N, k)
    y_pred = mode(y_pred.T).mode
    return accuracy_score(y, y_pred)


    



