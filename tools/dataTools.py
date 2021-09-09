import pathlib, pickle
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import qr, svd, inv
import pyaldata as pyal
from sklearn.decomposition import PCA
from typing import Callable

import logging


def summary(df):
    "prints a summary of monkey task datasets"
    print(df.monkey[0], df.date[0], '  ')
    time_sig = pyal.get_time_varying_fields(df)
    print(f'time signals:{time_sig}  ')
    for sig in time_sig:
        print(f'{sig} units: {df[sig][0].shape[1]}  ') if 'spike' in sig else 0
    try:
        print(f'tasks in file: {np.unique(df.task)}, epochs: {np.unique(df.epoch)}  ')
    except:
        print(f'tasks in file: {np.unique(df.task)}  ')
    try:
        print(f'Baseline trials: {np.sum(df.epoch=="BL")}  ')
    except:
        pass
    
    print('\n---\n')

def load_pyal_data(path: pathlib.Path) -> pd.DataFrame:
    """
    Loads pyal_data files. If it's a *.mat file, it is loaded, gets pickled and returned.
    If the pickle already exists, the pickle gets loaded instead of the *.mat file.
    
    Parameters
    ----------
    `path`: path to a *.mat file (or a *.p file).
    
    Returns
    -------
    `out`: a pyaldata pd.DataFrame
    """
    def load_pickle(pickle_path):
        with pickle_path.open('rb') as f:
            out = pickle.load(f)
            assert isinstance(out, pd.DataFrame), f'wrong data in pickle {pickle_path}'
        return out

    if isinstance(path, str):
        path = pathlib.Path(path)
    
    assert path.is_file(), f'path is not to a file: {path}'
    
    if path.suffix == '.p':
        return load_pickle(path)
    elif path.suffix == '.mat':
        pickle_path = path.with_suffix('.p')
        if pickle_path.exists():
            return load_pickle(pickle_path)
        else:
            df = pyal.mat2dataframe(path, shift_idx_fields=True)
            with open(pickle_path, 'wb') as f:
                pickle.dump(df, f)
            return df
    else:
        raise NameError(f'wrong file suffix: {path}')


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

    CC = canoncorr(d0, d1)

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

def VAF_pc_cc_pyal(df1:pd.DataFrame, field1: str, epoch1, target1: int,
                   df2:pd.DataFrame, field2: str, epoch2, target2: int,
                   n_components:int =10) -> (np.ndarray, np.ndarray):
    """
    Measure VAF for each CCA axis, between 2 DataFrames, fields, time epochs, and targets.
    epoch1, epoch2: an instance of the `pyal.generate_epoch_fun` function.
    """
    if "target_id" not in df1.columns:
        df1["target_id"] = df1.apply(get_target_id, axis=1)
    if "target_id" not in df2.columns:
        df2["target_id"] = df2.apply(get_target_id, axis=1)
 
    df1 = pyal.restrict_to_interval(df1,epoch_fun=epoch1)
    rates_1 = np.concatenate(df1[field1].values, axis=0)
    rates_1 -= np.mean(rates_1,axis=0)
    rates_1_model = PCA(n_components=n_components, svd_solver='full').fit(rates_1)
    rates_1_C = rates_1_model.components_
    df1 = pyal.apply_dim_reduce_model(df1, rates_1_model, field1, '_pca');

    
    df1 = pyal.select_trials(df1, df1.target_id==target1)
    pca_1_target = np.concatenate(df1['_pca'].values, axis=0)

    
    df2 = pyal.restrict_to_interval(df2, epoch_fun=epoch2)
    rates_2 = np.concatenate(df2[field2].values, axis=0)
    rates_2 -= np.mean(rates_2,axis=0)
    rates_2_model = PCA(n_components=n_components, svd_solver='full').fit(rates_2)
    rates_2_C = rates_2_model.components_
    df2 = pyal.apply_dim_reduce_model(df2, rates_2_model, field2, '_pca');
    
    df2 = pyal.select_trials(df2, df2.target_id==target2)
    pca_2_target = np.concatenate(df2['_pca'].values, axis=0)
    
    
    # same number of timepoints in both matrices
    n_samples = min ([pca_1_target.shape[0], pca_2_target.shape[0]])
    pca_1_target = pca_1_target[:n_samples,:]
    pca_2_target = pca_2_target[:n_samples,:]

    A, B, r, _, _ = canoncorr(pca_1_target, pca_2_target, fullReturn=True)
    VAFs1 = VAF_pc_cc(rates_1, rates_1_C, A)
    VAFs2 = VAF_pc_cc(rates_2, rates_2_C, B)
    
    return VAFs1, VAFs2, r

def VAF_pc_cc_pyal2(df1:pd.DataFrame, field1: str, epoch1, target1: int,
                    df2:pd.DataFrame, field2: str, epoch2, target2: int,
                    n_components:int =10, n_iter:int =20):
    """
    Identical to `VAF_pc_cc_pyal`, ...
    except that it  tries to correct for _very_ different number of units by...
    subsampling the larger population `n_iter` times and averaging over the results.
    """
    if "target_id" not in df1.columns:
        df1["target_id"] = df1.apply(get_target_id, axis=1)
    if "target_id" not in df2.columns:
        df2["target_id"] = df2.apply(get_target_id, axis=1)
 
    df1_ = pyal.restrict_to_interval(df1,epoch_fun=epoch1)
    rates_1 = np.concatenate(df1_[field1].values, axis=0)
    rates_1 -= np.mean(rates_1,axis=0)

    df2_ = pyal.restrict_to_interval(df2, epoch_fun=epoch2)
    rates_2 = np.concatenate(df2_[field2].values, axis=0)
    rates_2 -= np.mean(rates_2,axis=0)
    
    # PCA
    ## check for `n`
    n1 = rates_1.shape[1]
    n2 = rates_2.shape[1]
    n_s, n_l = min(n1, n2), max(n1, n2)
    
    if n1 >= n2:
        array1Bigger = True
    else:
        array1Bigger = False
    diffTooBig = (abs(n1-n2)/min(n1,n2) >= 2) or (abs(n1-n2) > 100)  # boolean

    rng = np.random.default_rng(12345)
    VAFs1, VAFs2, R = [], [], []
    if diffTooBig:
#         print(f'correcting for numbesr of units: {n1, n2}')
        for i in range(n_iter):
            idx = rng.choice(n_l, n_s)
            array_new = rates_1 if array1Bigger else rates_2
            array_new = array_new[:,idx]
            array_new_model = PCA(n_components=n_components, svd_solver='full').fit(array_new)
            
            if array1Bigger:
                PCA_model = PCA(n_components=n_components, svd_solver='full').fit(rates_2)
            else:
                PCA_model = PCA(n_components=n_components, svd_solver='full').fit(rates_1)

            rates_1_model = array_new_model if array1Bigger else PCA_model
            rates_1_C = rates_1_model.components_
            df1__ = pyal.select_trials(df1_, df1_.target_id==target1)
            rates_1_target = np.concatenate(df1__[field1].values, axis=0)
            pca_1_target = rates_1_model.transform(rates_1_target[:,idx]) if array1Bigger else rates_1_model.transform(rates_1_target)

            rates_2_model = PCA_model if array1Bigger else array_new_model
            rates_2_C = rates_2_model.components_
            df2__ = pyal.select_trials(df2_, df2_.target_id==target2)
            rates_2_target = np.concatenate(df2__[field2].values, axis=0)
            pca_2_target = rates_2_model.transform(rates_2_target) if array1Bigger else rates_2_model.transform(rates_2_target[:,idx])

            # same number of timepoints in both matrices
            n_samples = min ([pca_1_target.shape[0], pca_2_target.shape[0]])
            pca_1_target = pca_1_target[:n_samples,:]
            pca_2_target = pca_2_target[:n_samples,:]

            A, B, r, _, _ = canoncorr(pca_1_target, pca_2_target, fullReturn=True)
            V1 = VAF_pc_cc(array_new if array1Bigger else rates_1, rates_1_C, A)
            V2 = VAF_pc_cc(rates_2 if array1Bigger else array_new, rates_2_C, B)
            VAFs1.append(V1)
            VAFs2.append(V2)
            R.append(r)
            
        VAFs1 = np.mean(np.array(VAFs1), axis=0)
        VAFs2 = np.mean(np.array(VAFs2), axis=0)
        R = np.mean(np.array(R), axis=0)
    else:
        VAFs1,VAFs2,R = VAF_pc_cc_pyal(df1, field1, epoch1, target1,
                                       df2, field2, epoch2, target2, n_components)

    return VAFs1, VAFs2, R


def PCA_n_corrected(array1:np.ndarray, array2:np.ndarray, n_iter:int =20, n_components:int =10):
    """
    Repeat PCA `n_iter` times by subsampling the larger dataset. (not fully tested)

    Parameters
    ----------
    `array1` and `array2` are: time x units
    
    Returns
    -------
    `PCA_models1` and `PCA_models2`: lists containing PCA objects. If the number of units needs correction, lengths are bigger than 1.

    
    """
    n1 = array1.shape[1]
    n2 = array2.shape[1]
    n_s, n_l = min(n1, n2), max(n1, n2)
    
    if n1>=n2:
        array1Bigger = True
    else:
        array1Bigger = False
    
    diffTooBig = (abs(n1-n2)/min(n1,n2) >= 2) or (abs(n1-n2) > 100)  # boolean
    rng = np.random.default_rng(12345)

    PCA_models1, PCA_models2 = [], []
    if diffTooBig:
        PCA_models=[]
        for i in range(n_iter):
            idx = rng.choice(n_l, n_s)
            array_new = array1 if array1Bigger else array2
            array_new = array_new[:,idx]
            array_new_model = PCA(n_components=n_components, svd_solver='full').fit(array_new)
            PCA_models.append(array_new_model)

        if array1Bigger:
            PCA_models1 = PCA_models
            PCA_models2.append(PCA(n_components=n_components, svd_solver='full').fit(array2))
            PCA_models2 *= n_iter
        else:
            PCA_models2 = PCA_models
            PCA_models1.append(PCA(n_components=n_components, svd_solver='full').fit(array1))
            PCA_models1 *= n_iter

    
    else:
        PCA_models1.append(PCA(n_components=n_components, svd_solver='full').fit(array1))
        PCA_models2.append(PCA(n_components=n_components, svd_solver='full').fit(array2))

    return PCA_models1, PCA_models2


def get_data_array(data_list: list[pd.DataFrame], epoch: Callable =None , area: str ='M1', model: Callable =None, n_components:int = 10) -> np.ndarray:
    """
    Applies the `model` to the data and return a data matrix of the shape: sessions x targets x trials x time x modes
    with the minimum number of trials and timepoints shared across all the datasets/targets.
    
    Parameters
    ----------
    `data_list`: list of pd.dataFrame datasets from pyalData (could also be a single dataset)
    `epoch`: an epoch function of the type `pyal.generate_epoch_fun`
    `area`: area, either: 'M1', or 'S1', or 'PMd', ...
    `model`: a model that implements `.fit()`, `.transform()` and `n_components`. By default: `PCA(10)`. If it's an integer: `PCA(integer)`.
    `n_components`: use `model`, this is for backward compatibility
    
    Returns
    -------
    `AllData`: np.ndarray

    Signature
    -------
    AllData = get_data_array(data_list, execution_epoch, area='M1', model=10)
    all_data = np.reshape(AllData, (-1,10))
    """
    if isinstance(data_list, pd.DataFrame):
        data_list = [data_list]
    if model is None:
        model = PCA(n_components=n_components, svd_solver='full')
    elif isinstance(model, int):
        model = PCA(n_components=model, svd_solver='full')
    
    field = f'{area}_rates'
    n_shared_trial = np.inf
    target_ids = np.unique(data_list[0].target_id)
    for df in data_list:
        for target in target_ids:
            df_ = pyal.select_trials(df, df.target_id== target)
            n_shared_trial = np.min((df_.shape[0], n_shared_trial))

    n_shared_trial = int(n_shared_trial)

    # finding the number of timepoints
    if epoch is not None:
        df_ = pyal.restrict_to_interval(data_list[0],epoch_fun=epoch)
    n_timepoints = int(df_[field][0].shape[0])

    # pre-allocating the data matrix
    AllData = np.empty((len(data_list), len(target_ids), n_shared_trial, n_timepoints, model.n_components))

    rng = np.random.default_rng(12345)
    for session, df in enumerate(data_list):
        df_ = pyal.restrict_to_interval(df, epoch_fun=epoch) if epoch is not None else df
        rates = np.concatenate(df_[field].values, axis=0)
        rates -= np.mean(rates, axis=0)
        rates_model = PCA(n_components=n_components, svd_solver='full').fit(rates)
        df_ = pyal.apply_dim_reduce_model(df_, rates_model, field, '_pca');

        for targetIdx,target in enumerate(target_ids):
            df__ = pyal.select_trials(df_, df_.target_id==target)
            all_id = df__.trial_id.to_numpy()
            # to guarantee shuffled ids
            while ((all_id_sh := rng.permutation(all_id)) == all_id).all():
                continue
            all_id = all_id_sh
            # select the right number of trials to each target
            df__ = pyal.select_trials(df__, lambda trial: trial.trial_id in all_id[:n_shared_trial])
            for trial, trial_rates in enumerate(df__._pca):
                AllData[session,targetIdx,trial, :, :] = trial_rates

    return AllData

def add_history(data:np.ndarray, n_hist:int) -> np.ndarray:
    """
    Adds history to the columns of `data`, by stacking `n_hist` previous time bins

    Parameters
    ----------
    `data`: the data matrix, T x n with _T_ time points and _n_ neurons/components/features.
    
    `n_hist` : number of time rows to be added.
    
    Returns
    -------
    An array of _T_  x _(n x n_hist+1)_

    """
    out = np.hstack([np.roll(data, shift, axis=0) for shift in range(n_hist+1)])
    out[:n_hist,data.shape[1]:] = 0
    return out

def add_history_to_data_array(allData, n_hist):
    """
    applies `add_history` to each trial

    Parameters
    ----------
    `allData`: the data matrix coming from `dt.add_history`
    
    `n_hist` : number of time rows to be added.
    
    Returns
    -------
    Similar to the output of `dt.get_data_array`, with extra PC columns.
    """
    assert allData.ndim == 5, 'Wrong input size'
    newShape = list(allData.shape)
    newShape[-1] *= (n_hist+1)
    
    out = np.empty(newShape)
    for session,sessionData in enumerate(allData):
        for target,targetData in enumerate(sessionData):
            for trial,trialData in enumerate(targetData):
                out[session,target,trial,:,:] = add_history(trialData, n_hist)
    return out
