import pandas as pd
import numpy as np

def process_data(fname):
    """
        input: file name for working datasets
        output: X, y dataset
        --
        read the datasets
    """
    data = pd.read_csv(fname, sep="\t")
    X = data.iloc[:, 2:].as_matrix()
    y = data.iloc[:,1].as_matrix()
    return (X, y)

def psedo_data(X_ori, y_ori, seed=0):
    """
        input: oringal n of X, y
        output: total 2n of X, y
                number of true SNP
        ---
        generate shadow SNP
        by random generator
    """
    np.random.seed(seed)
    X_gen = []
    for i in range(len(X_ori[0])):
        temp_X = np.random.permutation(X_ori[:,i])
        X_gen.append(temp_X)
    X_gen = np.swapaxes(np.array(X_gen),0,1) 
    X = np.hstack((X_ori, X_gen))
    return (X, y_ori)

def readdata(fname, seed=0):
    X_ori, y_ori = process_data(fname)
    _, n_snp = X_ori.shape
    X, y = psedo_data(X_ori, y_ori, seed)
    return (np.array(X), np.array(y), n_snp)

