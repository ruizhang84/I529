import numpy as np
from readsnp import readdata
from sklearn.feature_selection import chi2

def split_snp_chi2(X, y, snp, thresh=0.05):
    """
        input:  X, y datasets, informative SNPs indx
                partition on weak and strong by means
        output: two groups features
    """
    def split(pval, snp_idx, thresh):
        """
            split index into 2 groups
        """
        weak_idx = []
        strong_idx = []

        for i in range(len(pval)):
            if pval[i] <= thresh:
                strong_idx.append(snp_idx[i])
            else:
                weak_idx.append(snp_idx[i])
        return (weak_idx, strong_idx)

    X_test = X[:, snp]
    _, pval= chi2(X_test, y)

    weak, strong = split(pval, snp, thresh)
    return (weak, strong)
