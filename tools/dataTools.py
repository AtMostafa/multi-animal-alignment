from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import qr, svd, inv

import logging


def canoncorr(X:np.array, Y: np.array, fullReturn: bool = False) -> np.array:
    """
    Canonical Correlation Analysis (CCA)
    line-by-line port from Matlab implementation of `canoncorr`
    X,Y: (samples/observations) x (features) matrix, for both: X.shape[0] >> X.shape[1]
    fullReturn: whether all outputs should be returned or just `r` be returned (not in Matlab)
    
    returns: A,B,r,U,V 
    A,B: Canonical coefficients for X and Y
    U,V: Canonical scores for the variables X and Y
    r:   Canonical correlations
    
    Signature:
    A,B,r,U,V = canoncorr(X, Y)
    """
    n, p1 = X.shape
    p2 = Y.shape[1]
    if p1 >= n or p2 >= n:
        logging.warning('Not enough samples, might cause problems')

    # Center the variables
    X = X - np.mean(X,0);
    Y = Y - np.mean(Y,0);

    # Factor the inputs, and find a full rank set of columns if necessary
    Q1,T11,perm1 = qr(X, mode='economic', pivoting=True, check_finite=True)

    rankX = sum(np.abs(np.diagonal(T11)) > np.finfo(type((np.abs(T11[0,0])))).eps*max([n,p1]));

    if rankX == 0:
        logging.error(f'stats:canoncorr:BadData = X')
    elif rankX < p1:
        logging.warning('stats:canoncorr:NotFullRank = X')
        Q1 = Q1[:,:rankX]
        T11 = T11[rankX,:rankX]

    Q2,T22,perm2 = qr(Y, mode='economic', pivoting=True, check_finite=True)
    rankY = sum(np.abs(np.diagonal(T22)) > np.finfo(type((np.abs(T22[0,0])))).eps*max([n,p2]));

    if rankY == 0:
        logging.error(f'stats:canoncorr:BadData = Y')
    elif rankY < p2:
        logging.warning('stats:canoncorr:NotFullRank = Y')
        Q2 = Q2[:,:rankY];
        T22 = T22[:rankY,:rankY];

    # Compute canonical coefficients and canonical correlations.  For rankX >
    # rankY, the economy-size version ignores the extra columns in L and rows
    # in D. For rankX < rankY, need to ignore extra columns in M and D
    # explicitly. Normalize A and B to give U and V unit variance.
    d = min(rankX,rankY);
    L,D,M = svd(Q1.T @ Q2, full_matrices=True, check_finite=True, lapack_driver='gesdd')
    M = M.T

    A = inv(T11) @ L[:,:d] * np.sqrt(n-1);
    B = inv(T22) @ M[:,:d] * np.sqrt(n-1);
    r = D[:d]
    # remove roundoff errs
    r[r>=1] = 1
    r[r<=0] = 0

    if not fullReturn:
        return r

    # Put coefficients back to their full size and their correct order
    A[perm1,:] = np.vstack((A, np.zeros((p1-rankX,d))))
    B[perm2,:] = np.vstack((B, np.zeros((p2-rankY,d))))
    
    # Compute the canonical variates
    U = X @ A
    V = Y @ B

    return A, B, r, U, V

def CCA_pyal(df1:pd.DataFrame, field1: str, df2:pd.DataFrame =None, field2:str =None) -> np.array:
    """
    Rerurns CCs between 2 fields from 2 pyalData dataframes.
    If, `df2` is not specified, then `field2` must be valid and canoncorr will be applied on
    2 fields in `df1`
    
    Returns CC, i.e., ccacnonical correlations
    """
    if df2 is None:
        assert isinstance(field2,str), 'Enter a valid string in field2'
        df2 = df1

    d0 = np.concatenate(df1[field1].values, axis=0)
    d1 = np.concatenate(df2[field2].values, axis=0)

    # same number of timepoints in both matrices
    n_samples = min ([d0.shape[0], d1.shape[0]])
    d0 = d0[:n_samples,:]
    d1 = d1[:n_samples,:]

    CC = dt.canoncorr(d0, d1)

    return CC

def VAF_pc_cc (X: np.ndarray, C: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Calculate Variance Accounted For (VAF) for a double projection (as in from PCA --> to CCA) using the method in Gallego, NatComm, 2018
    
    Parameters
    ----------
    `X`: the data matrix, T x n with _T_ time points and _n_ neurons, and each neuron is **zero mean**.
    
    `C`: the first projection matrix, usually it is the `PCA_model.components_`, but in principle could be any projection matrix with orthogonal bases.
    
    `A` : is the CCA canonical axes, the output of the `canoncorr` function, in principle could be any projection matrix, not necessarily orthogonal.
    
    Returns
    -------
    `VAFs`: np.array with VAF for each axes of `C`, normalised between 0<VAF<1 for each axis, `sum(VAFs)` equals to total VAF.
    """
    # following the notation in Gallego 2018
    D = inv(A.T@A)@A.T@C
    E = C.T@A
    norm = lambda m:np.sum(m**2)
    
    VAFs=np.empty((C.shape[0],))
    for comp in range(1,C.shape[0]+1):
        VAF = norm(X - X @ E[:,:comp] @ D[:comp,:]) / norm(X)
        VAFs[comp-1] = 1-VAF

    VAFs = np.array([VAFs[0],*np.diff(VAFs)])
    return VAFs
