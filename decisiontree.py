import numpy as np
import random

def selectSNP(weak, strong, mtry=50):
    """
        At each node, select a subspace of mtry
    """ 
    mtryw = mtry*len(weak)//(len(weak)+len(strong))
    mtrys = mtry*len(strong)//(len(weak)+len(strong))
    # select SNPs
    if mtryw > len(weak):
        weak_idx = weak
    else:
        weak_idx = np.random.choice(weak, mtryw, replace=False)
    if mtrys > len(strong):
        strong_idx = strong
    else:
        strong_idx = np.random.choice(strong, mtrys, replace=False)
    idx = np.concatenate([weak_idx, strong_idx])
    return idx

def calcGini(y):
    """
        input:  X, y datasets
                index for the selected features
        output: gini score
        --
        compute the gini score
    """
    if len(y) == 0:
        return 0.0

    p1 = np.sum(y)*1.0/len(y)
    p2 = 1.0 - p1
    gini = 1.0 - p1**2 - p2**2
    return gini

def chooseBestFeatureToSplit(X, y, strong, weak, mtry):
    """
        randomly generate some pairs to try
    """
    features = selectSNP(weak, strong, mtry)
    bestGini = float("inf")
    bestFeature = 0

    for idx in features:
        y_0 = y[X[:, idx] == 0]
        y_1 = y[X[:, idx] != 0]
        splitGini = (len(y_0)*calcGini(y_0)/len(y)
                    +len(y_1)*calcGini(y_1)/len(y))
        if splitGini < bestGini:
            bestGini = splitGini
            bestFeature = idx

    strong = [i for i in strong if i != bestFeature]
    weak = [i for i in weak if i != bestFeature]
    return (strong, weak, bestFeature)

def splitDataSet(X, y, bestFeature):
    X_0 = X[X[:, bestFeature] == 0]
    X_1 = X[X[:, bestFeature] != 0]
    y_0 = y[X[:, bestFeature] == 0]
    y_1 = y[X[:, bestFeature] != 0]
    return (X_0, y_0, X_1, y_1)

class Node:
    """
        a node in decision tree
    """
    def __init__(self):
        self.left = None
        self.right = None
        self.bestFeature = None
    
    def predict(self, X_i):
        if self.left is None:
            return self.label
        else:
            feat = X_i[self.bestFeature]
            if feat < 1:
                return self.left.predict(X_i)
            else:
                return self.right.predict(X_i)

def decision(y):
    return np.around(np.mean(y))

def build_tree(X, y, strong, weak, mtry, nmin):
    """
        build decision tree
    """
    root = Node()
    # not enough features
    if len(strong) + len(weak) <= nmin or len(strong) == 0 or len(weak) == 0:
        root.label = decision(y)
        return root
    
    # pure y label
    if np.sum(y) == len(y):
        root.label = decision(y)
        return root
    elif np.sum(y) == 0:
        root.label = decision(y)
        return root

    strong, weak, idx = chooseBestFeatureToSplit(X, y, strong, weak, mtry)
    X_lf, y_lf, X_rt, y_rt = splitDataSet(X, y, idx)
    root.bestFeature = idx

    # pure x datasets
    if len(X_lf) == 0 or len(X_lf) == len(X):
        root.label = decision(y)
        return root

    root.left = build_tree(X_lf, y_lf, strong, weak, mtry, nmin)
    root.right = build_tree(X_rt, y_rt, strong, weak, mtry, nmin)
    return root

class Classifer:
    """
        decision tree
    """
    def __init__(self, weak=[], strong=[], mtry=50, nmin=1):
        self.weak = weak
        self.strong = strong
        self.mtry = mtry
        self.nmin = nmin
        self.root = Node()

    def fit(self, X, y):
        """
            training the datasets
        """
        self.root = build_tree( X, y, self.strong, self.weak, self.mtry, self.nmin)

    def predict(self, X):
        pred = []
        for X_i in X:
            pred.append(self.root.predict(X_i))
        return np.array(pred)