from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor
import torch



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

def knn_class_score(model:torch.nn.Module, x_train:Tensor, x_test:Tensor, y_train, y_test, k=1, device='cpu'):
    '''subset accuracy of labeling using the nearest projected neighbor'''
    model.eval()
    model = model.to(device)
    x_train = x_train.to(device)
    x_test = x_test.to(device)
    with torch.no_grad():
        emb_train = model.forward(x_train).cpu()
        emb_test = model.forward(x_test).cpu()
    knn = KNeighborsClassifier(k)
    knn.fit(emb_train, y_train)
    return knn.score(emb_test, y_test)
