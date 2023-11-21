from sklearn.neighbors import KNeighborsClassifier, KNeighborsTransformer, sort_graph_by_row_values
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix, find
from scipy.stats import mode
from torch import Tensor
import torch
from models import Model
import numpy as np



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

def knn_ref_score(model:Model, x_train:Tensor, x_test:Tensor, y_train, y_test, k=3, device='cpu'):
    '''
    Compute subset accuracy of knn classification, using x_train embeddings as the reference and x_test as the example to classify
    '''
    model.eval()
    model = model.to(device)
    x_train = x_train.to(device)
    x_test = x_test.to(device)
    with torch.no_grad():
        emb_train = model.embed(x_train).cpu()
        emb_test = model.embed(x_test).cpu()
    knn = KNeighborsClassifier(k)
    knn.fit(emb_train, y_train)
    return knn.score(emb_test, y_test)

@torch.no_grad
def knn_self_score(embed:Tensor, labels:Tensor,k=3):
    '''
    Compute subset accuracy of knn classification, using x_j (j != i) to classify each x_i
    '''
    knn = KNeighborsTransformer(n_neighbors=k)
    labels = labels.cpu().numpy()
    knn.fit(embed.cpu(), )
    graph = knn.kneighbors_graph()
    labels_pred = labels[find(graph)[1]] # neighbors' labels (find(graph)[1] are the target nodes in the knn graph)
    labels_pred = labels_pred.reshape((-1, k)) # (kN, ) -> (N, k)
    labels_pred = mode(labels_pred.T).mode
    return accuracy_score(labels, labels_pred)


    



