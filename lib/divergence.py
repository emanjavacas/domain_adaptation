import numpy as np
from scipy.spatial import cKDTree as KDTree


def KLdiv(X1, X2):
    "computes KL divergence for two example matrices"
    P = np.array([X1[:, i].mean()   # feature mean
                  + .5/X1.shape[0]   # smoothed and averaged over examples
                  for i in X1.shape[1]])
    Q = np.array([X2[:, i].mean()
                  + .5/X2.shape[0]
                  for i in X2.shape[1]])
    div = sum(np.multiply(P, np.log(P/Q)))
    return div


def KLdivTree(X1, X2):
    "fast KL estimation using KDTrees"
    n, d = X1.shape
    m, dy = X2.shape
    xtree = KDTree(X1)
    ytree = KDTree(X2)
    r = xtree.query(X1, k=2, eps=.01, p=2)[0][:, 1]
    s = ytree.query(X1, k=1, eps=.01, p=2)[0]
    diff = r/s
    return -np.log(diff).sum() * d / n + np.log(m/(n-1))


def Sw(X1, y):
    "within-class scatter"
    sw = 0
    classes = np.unique(y)
    for i in range(len(classes)):
        idx = np.squeeze(np.where(y == classes[i]))
        d = np.squeeze(X1[idx, :])
        classcov = np.cov(np.transpose(d))
        n = np.shape(idx)[0]
        py = float(n / X1.shape[1])
        sw += float(py * classcov)
    return sw
