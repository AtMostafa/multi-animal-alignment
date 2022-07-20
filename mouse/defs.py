import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import PCA
import scipy.linalg as linalg
import scipy.stats as stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, r2_score
import pyaldata as pyal

rng = np.random.default_rng(np.random.SeedSequence(12345))

_example = ('js2p0_tbytSpkHandJsTrj10msBin_WR38_052219_ss.p', 'js2p0_tbytSpkHandJsTrj10msBin_WR40_081919_ss.p')
MAX_HISTORY = 3  #int: no of bins to be added as history
BIN_SIZE = .03  # sec
WINDOW_ctrl = (-.95, -.5)
WINDOW_prep = (-.4, .05)  # sec
WINDOW_exec = (-.05, .4)  # sec
n_components = 10  # min between M1 and Str
areas = ('M1', 'Str')
n_targets = 2

prep_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on',
                                     rel_start=int(WINDOW_prep[0]/BIN_SIZE),
                                     rel_end=int(WINDOW_prep[1]/BIN_SIZE)
                                    )
exec_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on',
                                     rel_start=int(WINDOW_exec[0]/BIN_SIZE),
                                     rel_end=int(WINDOW_exec[1]/BIN_SIZE)
                                    )
fixation_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on',
                                         rel_start=int(WINDOW_ctrl[0]/BIN_SIZE),
                                         rel_end=int(WINDOW_ctrl[1]/BIN_SIZE)
                                        )
exec_epoch_decode = pyal.generate_epoch_fun(start_point_name='idx_movement_on',
                                     rel_start=int(WINDOW_exec[0]/BIN_SIZE) - MAX_HISTORY,
                                     rel_end=int(WINDOW_exec[1]/BIN_SIZE)
                                    )

def custom_r2_func(y_true, y_pred):
    "$R^2$ value as squared correlation coefficient, as per Gallego, NN 2020"
    c = np.corrcoef(y_true.T, y_pred.T) ** 2
    return np.diag(c[-int(c.shape[0]/2):,:int(c.shape[1]/2)])

custom_r2_scorer = make_scorer(custom_r2_func)

def prep_general_mouse (df):
    "preprocessing general! for J. Dudman mouse data"
    # rename unit fields
    old_fields = [col for col in df.columns.values if 'unit' in col]
    new_fields = ['M1_spikes' if 'Ctx' in col else 'Str_spikes' for col in old_fields]
    df_ = df.rename(columns = {old:new for old,new in zip(old_fields,new_fields)})
    # change spikes datatype
    for signal in new_fields:
        df_[signal] = [np.nan_to_num(x=s.toarray().T, nan=0) for s in df_[signal]]
    # add trial_id
    df_['trial_id'] = np.arange(1,df_.shape[0]+1)
    # only keep good trials
    df_= pyal.select_trials(df_, df_.trialType== 'sp')
    # fill no-laser trials (and index fields) with zero
    n_bins = df_[new_fields[0]][0].shape[0]
    var_len_fields = [ 'spkPullIdx', 'spkRchIdx', 'spkTimeBlaserI']
    fill_zeros = lambda a: a if len(a)>1 else np.zeros((n_bins,))
    for field in var_len_fields:
        if field not in df_.columns:continue
        df_[field] = [fill_zeros(s) for s in df_[field]]
    # fill fields that are cut with np.nans and remove trials that are too long or don't exist
    cut_fields = ['hTrjB', 'hVelB','hDistFromInitPos']
    df_['badIndex'] = [max(trialT.shape)>n_bins or
                       max(trialV.shape)>n_bins or
                       max(trialD.shape)>n_bins or
                       max(trialT.shape) < 2 or 
                       max(trialV.shape) < 2 or 
                       np.isnan(trialT).sum() > 5 for trialT,trialV,trialD in zip(df_.hTrjB,df_.hVelB,df_.hDistFromInitPos)]
    df_= pyal.select_trials(df_, df_.badIndex == False)
    df_.drop('badIndex', axis=1, inplace=True)
    fill_nans = lambda a: a if max(a.shape)==n_bins else np.pad(a, (((0,n_bins-max(a.shape)),)+(len(a.shape)-1)*((0,0),)), 'constant', constant_values=(np.nan,))
    for field in cut_fields:
        if field not in df_.columns:continue
        df_[field] = [fill_nans(s.T) for s in df_[field]]   
    # add bin_size
    df_['bin_size']=0.01  # data has 10ms bin size
    # add idx_movement_on which is exactly at t=df.timeAlign
    df_['idx_movement_on'] = [np.argmin(np.abs(s-i)) for i,s in zip(df_['timeAlign'],df_['spkTimeBins'])]
    # add pull start idx
    df_['idx_pull_on'] = [pullIdx.nonzero()[0][0] if len(pullIdx.nonzero()[0])>0 else np.nan for pullIdx in df_.spkPullIdx]
    # add pull stop idx
    df_['idx_pull_off'] = [min((pull.nonzero()[0][-1], velNans[0] if len(velNans:=np.isnan(vel).nonzero()[0])>0 else [np.inf])) for pull,vel in zip(df_.spkPullIdx,df_.hVelB)]
    # remove trials with no pull idx
    df_.dropna(subset=['idx_pull_on'], inplace=True)
    df_.idx_pull_on = df_.idx_pull_on.astype(np.int32)
    df_.index = np.arange(df_.shape[0])
    # add target_id
    rem = np.remainder(df_['blNumber'].to_numpy(), 4)
    rem[np.logical_or(rem==3 , rem ==0)] = 0
    rem[np.logical_or(rem==1 , rem==2)] = 1
    df_['target_id'] = rem

    for signal in new_fields:
        df_ = pyal.remove_low_firing_neurons(df_, signal, 1)

    df_ = pyal.select_trials(df_, df_.idx_movement_on < df_.idx_pull_on)
    df_ = pyal.select_trials(df_, df_.idx_pull_on < df_.idx_pull_off)
    # !!! discard outlier behaviour---tricky stuff !!!
        # reach duration < 500ms
    df_ = pyal.select_trials(df_, df_.idx_pull_on - df_.idx_movement_on < 50)
        # pull duration < 450ms
    df_ = pyal.select_trials(df_, df_.idx_pull_off - df_.idx_pull_on < 45)

    try:
        noLaserIndex = [i for i,laserData in enumerate(df_.spkTimeBlaserI) if not np.any(laserData)]
        df_= pyal.select_trials(df_, noLaserIndex)
    except AttributeError:
        # due to absence of this field in no-laser sessions
        pass

    df_ = pyal.combine_time_bins(df_, int(BIN_SIZE/.01))
    for signal in new_fields:
        df_ = pyal.sqrt_transform_signal(df_, signal)

    df_= pyal.add_firing_rates(df_, 'smooth', std=0.05)

    return df_

def get_data_array_and_vel(data_list: list[pd.DataFrame], epoch , area: str ='M1', n_components: int =10) -> np.ndarray:
    """
    Applies PCA to the data and return a data matrix of the shape: sessions x targets x  trials x time x PCs
    with the minimum number of trials and timepoints shared across all the datasets/targets.
    
    Parameters
    ----------
    `data_list`: list of pd.dataFrame datasets from pyal-data
    `epoch`: an epoch function of the type `pyal.generate_epoch_fun`
    `area`: area, either: 'M1', or 'S1', or 'PMd'

    Returns
    -------
    `AllData`: np.array

    Signature
    -------
    AllData = get_data_array(data_list, execution_epoch, area='M1', n_components=10)
    all_data = np.reshape(AllData, (-1,10))
    """
    field = f'{area}_rates'
    n_shared_trial = np.inf
    for df in data_list:
        for target in range(n_targets):
            df_ = pyal.select_trials(df, df.target_id== target)
            n_shared_trial = np.min((df_.shape[0], n_shared_trial))

    n_shared_trial = int(n_shared_trial)

    # finding the number of timepoints
    df_ = pyal.restrict_to_interval(df_,epoch_fun=epoch)
    n_timepoints = int(df_[field][0].shape[0])

    # pre-allocating the data matrix
    AllData = np.empty((len(data_list), n_targets, n_shared_trial, n_timepoints, n_components))
    AllVel  = np.empty((len(data_list), n_targets, n_shared_trial, n_timepoints, 3))
    for session, df in enumerate(data_list):
        df_ = pyal.restrict_to_interval(df, epoch_fun=epoch)
        rates = np.concatenate(df_[field].values, axis=0)
        rates_model = PCA(n_components=n_components, svd_solver='full').fit(rates)
        df_ = pyal.apply_dim_reduce_model(df_, rates_model, field, '_pca');

        for target in range(n_targets):
            df__ = pyal.select_trials(df_, df_.target_id==target)
            all_id = df__.trial_id.to_numpy()
            rng.shuffle(all_id)
            # select the right number of trials to each target
            df__ = pyal.select_trials(df__, lambda trial: trial.trial_id in all_id[:n_shared_trial])
            for trial, (trial_rates,trial_vel) in enumerate(zip(df__._pca, df__.hVelB)):
                AllData[session,target,trial, :, :] = trial_rates
                AllVel[session,target,trial, :, :] = trial_vel
    
    return AllData, AllVel

def time_trim(a,b):
    l = min(a.shape[0],b.shape[0])
    return a[:l],b[:l]
