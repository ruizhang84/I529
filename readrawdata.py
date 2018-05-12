import pandas as pd
import numpy as np
import time

DNA = ['A', 'T', 'C', 'G']

def argmax_dna(snp_obs):
    """
        input: a snp on all observations
        output: the major elle
    """
    count = np.zeros(4)
    for i in range(len(DNA)):
        count[i] = np.sum(snp_obs==DNA[i])
    return DNA[np.argmax(count)]

def process_snp(obs):
    """
        input: observation shape (n_snp, n_observations)
        output: X datasets
                snp with 0, 1, 2
        ---
        given a snp ATCG on each observaton
        convert snp in numberic representation on major and minor
    """
    n_snp, n_obs = obs.shape
    X = []
    for i in range(n_snp):
        # all obsevations on a snp
        snp_obs = obs[i]
        snp_r = np.ones(n_obs)*2
        snp_r[snp_obs==argmax_dna(snp_obs)] = 0
        X.append(snp_r)
    
    X = np.swapaxes(np.array(X),0,1)
    return X


def process_Xdata(fname):
    """
        input: file name for raw data
        output: X dataset
        --
        read the datasets
    """
    data = pd.read_csv(fname, sep=",")


    obs = data.iloc[:, 2:]
    # reorder col name
    col_name = list(obs)
    obs = obs[sorted(col_name, key=float)]

    snp_pos = data.iloc[:,:2]
    X = process_snp(obs.as_matrix())
    return (X, snp_pos)
    

def process_Ydata(fname):
    """
        input: file name for phenotype
        output: y dataset
        --
        read the datasets
    """   
    data = pd.read_csv(fname, sep=",")
    y_data = data['44_FRI']

    null_pos = y_data.isnull()
    y = y_data > np.mean(y_data)
    return (y, null_pos)

def filter_nan(X, y, null_pos):
    """
        input: X, y, nan values row index
        output: X, y by removing nan values 
    """
    mask = np.array(np.logical_not(null_pos))
    X = X[mask]
    y = y[mask]
    return (X, y)

def pseudo_data(X_ori, y_ori, seed=0):
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

def readdata(genotype_fname, phenotype_fname, seed=0):
    """
        input: file name for working datasets
        output: X, y dataset
        --
        read the datasets
    """
    X, snp_pos = process_Xdata(genotype_fname)
    snp_pos.to_csv("../Data/snp_chromos_position.csv")
    y, null_pos = process_Ydata(phenotype_fname)
    X, y = filter_nan(X, y, null_pos)

    _, n_snp = X.shape
    X, y = pseudo_data(X, y, seed)
    return (X, y, n_snp)


