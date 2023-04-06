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
import math
from typing import Callable


rng = np.random.default_rng(np.random.SeedSequence(12345))
ref_file = 'Chewie_RT_CS_2016-10-21.mat'
ex_file = 'Mihili_RT_VR_2014-01-14.mat'
# raster_example = ('Chewie_CO_FF_2016-10-13.p', 'Mihili_CO_VR_2014-03-03.p')
# MAX_HISTORY = 3  #int: no of bins to be added as history
BIN_SIZE = .03  # sec
# WINDOW_prep = (-.4, .05)  # sec
WINDOW_exec = (0.0, .35)  # sec
n_components = 10  # min between M1 and PMd
areas = ('M1', 'PMd', 'MCx')
n_angle_groups = 6
subset_radius = 2
edge_dist = 3.5

target_grid = (3,3)
match_mse_cutoff_perc = 2
n_centers = target_grid[0]*target_grid[1]
target_groups = np.array([str(i)+ '_'+ str(j) for i in range(n_centers) for j in range(n_angle_groups)])

min_trials_per_target = 6
# prep_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on',
#                                      rel_start=int(WINDOW_prep[0]/BIN_SIZE),
#                                      rel_end=int(WINDOW_prep[1]/BIN_SIZE)
#                                     )
exec_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on',
                                     rel_start=int(WINDOW_exec[0]/BIN_SIZE),
                                     rel_end=int(WINDOW_exec[1]/BIN_SIZE)
                                    )
# fixation_epoch = pyal.generate_epoch_fun(start_point_name='idx_target_on',
#                                          rel_start=int(WINDOW_prep[0]/BIN_SIZE),
#                                          rel_end=int(WINDOW_prep[1]/BIN_SIZE)
#                                         )
# exec_epoch_decode = pyal.generate_epoch_fun(start_point_name='idx_movement_on',
#                                      rel_start=int(WINDOW_exec[0]/BIN_SIZE) - MAX_HISTORY,
#                                      rel_end=int(WINDOW_exec[1]/BIN_SIZE)
#                                     )

# def get_target_id(trial):
#     return int(np.round((trial.target_direction + np.pi) / (0.25*np.pi))) - 1

def prep_general (df):
    "preprocessing general!"
    time_signals = [signal for signal in pyal.get_time_varying_fields(df) if 'spikes' in signal]
    # df["target_id"] = df.apply(get_target_id, axis=1)  # add a field `target_id` with int values
    df['session'] = df.monkey[0]+':'+df.date[0]
    
    for signal in time_signals:
        df_ = pyal.remove_low_firing_neurons(df, signal, 1)
    
    MCx_signals = [signal for signal in time_signals if 'M1' in signal or 'PMd' in signal]
    if len(MCx_signals) > 1:
        df_ = pyal.merge_signals(df_, MCx_signals, 'MCx_spikes')
    elif len(MCx_signals) == 1:
        df_ = pyal.rename_fields(df_, {MCx_signals[0]:'MCx_spikes'})
    time_signals = [signal for signal in pyal.get_time_varying_fields(df_) if 'spikes' in signal]

    df_= pyal.select_trials(df_, df_.result== 'R')
    df_= pyal.select_trials(df_, df_.epoch=='BL')
    
    assert np.all(df_.bin_size == .01), 'bin size is not consistent!'
    df_ = pyal.combine_time_bins(df_, int(BIN_SIZE/.01))
    for signal in time_signals:
        df_ = pyal.sqrt_transform_signal(df_, signal)
        
    df_= pyal.add_firing_rates(df_, 'smooth', std=0.05)

    #fix target locations
    df_['target_center'] = [np.reshape(x.T.flatten(), x.shape, order = 'C') for x in df_['target_center']]
    
    return df_

def get_reaches_df(df):

    df_reaches = pd.DataFrame()
    df[['idx_go_cue1',
        'idx_go_cue2',
        'idx_go_cue3',
        'idx_go_cue4']] = pd.DataFrame(df.idx_go_cue.tolist(), index= df.index)

    #separate by reaches
    for i in range(4):
        start_point_name = 'idx_go_cue' +str(i+1)
        end_point_name = ('idx_go_cue' + str(i+2)) if i < 3 else 'idx_trial_end'
        df_ = pyal.restrict_to_interval(df, start_point_name, end_point_name)
        if i>0:
            df_['start_center'] = [x[i-1] for x in df_['target_center']] #assume reach starts in target of prev reach
        df_['target_center'] = [x[i] for x in df_['target_center']]
        df_['reach'] = i+1
        df_['idx_go_cue']= df_[start_point_name]
        df_['idx_reach_end'] = df_[end_point_name] - df_[start_point_name] 
        df_reaches = pd.concat([df_reaches, df_])
    df_reaches = df_reaches.sort_values(by=['trial_id', 'reach'])
    # df_reaches = df_reaches[df_reaches.idx_reach_end < 200] #TODO: better cutoff for slow reaches

    #fix workspace center offset
    df_second_reaches = df_reaches[df_reaches.reach == 2] #look at offset for second reach
    idx = [all(np.abs(x) < 100) for x in df_second_reaches.start_center] #remove reaches with wrong centers
    df_second_reaches = df_second_reaches[idx]
    mean_offset = np.mean([pos[0] - start_center for start_center,pos in zip(df_second_reaches.start_center, df_second_reaches.pos)],axis = 0) 
    df_reaches['pos'] = [x - mean_offset for x in df_reaches['pos']] 

    #separate into target groups
    df_reaches = set_target_groups(df_reaches)
    return df_reaches
    
def set_target_groups(df):
    from scipy.spatial.distance import cdist

    #set centers for reaches to start from
    grid = target_grid
    xlim = (-10,10)
    ylim = (-10,10)

    x_centers = np.linspace(xlim[0]+edge_dist, xlim[1]-edge_dist, grid[0])
    y_centers = np.linspace(ylim[0]+edge_dist, ylim[1]-edge_dist, grid[1])
    x_centers2 = x_centers[:-1] + (x_centers[1]-x_centers[0])/2
    y_centers2 = y_centers[:-1] + (y_centers[1]-y_centers[0])/2
    centers = []
    for x in x_centers:
        for y in y_centers:
            centers.append([x,y])

    for x in x_centers2:
        for y in y_centers2:
            centers.append([x,y])

    centers = np.array(centers)

    #center targets and pos at origin
    first_pos = [x[0] for x in df.pos]
    dist_from_centers = cdist(centers,first_pos)
    
    df['center_dist'] = np.min(dist_from_centers,axis = 0)
    df['center_id'] = np.argmin(dist_from_centers,axis = 0)
    df['center']= [centers[x] for x in df.center_id]
    df['pos_centered'] = [pos - (pos[0]-center) for pos, center in zip(df.pos, df.center)]
    df['target_centered'] = [target - (pos[0]-center) for target,pos,center in zip(df.target_center, df.pos, df.center)]

    # get target angle and group
    df['target_angle'] = [math.degrees(math.atan2(target[1]-center[1],target[0]-center[0])) for target,center in zip(df.target_center, df.center)]
    df['target_angle'] = [360+x if x < 0 else x for x in df.target_angle]
    df['angle_group'] = [math.floor(x/(360/n_angle_groups)) for x in df.target_angle]
    df['target_group'] = [str(center)+'_'+str(angle) for center,angle in zip(df.center_id, df.angle_group)]

    return df

def get_matched_reaches_idx(df1, df2):
    from scipy.optimize import linear_sum_assignment

    # match by pos mse
    mses = np.zeros([len(df1), len(df2)])
    for i, pos1 in enumerate(df1.pos_centered):
        for j, pos2 in enumerate(df2.pos_centered):
            mse = np.mean((pos1-pos2)**2)
            mses[i][j] = mse

    row_idx, col_idx = linear_sum_assignment(mses.T, maximize=False)
    min_mses = mses.T[row_idx, col_idx]
    df1_idx = col_idx

    #remove trials with high mse
    all_mses= mses.T.flatten()
    # plt.hist(min_mses, histtype='step')
    cutoff = np.percentile(mses.T.flatten(), match_mse_cutoff_perc)
    keep_trials = (min_mses<cutoff)
    # print('Cutoff', cutoff, len(min_mses), sum(keep_trials))

    #get indices for matched trials
    df1_idx = df1_idx[keep_trials]
    df2_idx = np.array(range(len(df2)))[keep_trials]
    
    return df1_idx, df2_idx


def get_paired_data_arrays(df1, df2, epoch: Callable =None , area: str ='M1', model: Callable =None, n_components:int = 10) -> np.ndarray:
    """
    Applies the `model` to the `data_list` and return a data matrix of the shape: sessions x targets x trials x time x modes
    with the minimum number of trials and timepoints shared across all the datasets/targets.
    
    Parameters
    ----------
    `data_list`: list of pd.dataFrame datasets from pyalData (could also be a single dataset)
    `epoch`: an epoch function of the type `pyal.generate_epoch_fun()`
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
    assert (len(df1)== len(df2))

    if model is None:
        model = PCA(n_components=n_components, svd_solver='full')
    elif isinstance(model, int):
        model = PCA(n_components=model, svd_solver='full')
    
    field = f'{area}_rates'
    #get min trials for targets
    n_shared_trial = np.inf
    target_ids = np.unique(df1.target_id)

    for df in [df1,df2]:
        for target in target_ids:
            df_ = pyal.select_trials(df, df.target_id== target)
            n_shared_trial = np.min((df_.shape[0], n_shared_trial))

    n_shared_trial = int(n_shared_trial)

    # finding the number of timepoints
    if epoch is not None:
        df_ = pyal.restrict_to_interval(df1,epoch_fun=epoch)
    n_timepoints = int(df_[field][0].shape[0])

    # pre-allocating the data matrix
    AllData1 = np.empty((1, len(target_ids), n_shared_trial, n_timepoints, model.n_components))
    AllData2 = np.empty((1, len(target_ids), n_shared_trial, n_timepoints, model.n_components))

    rng = np.random.default_rng(12345)

    df1_ = pyal.restrict_to_interval(df1, epoch_fun=epoch) if epoch is not None else df
    df1_ = pyal.dim_reduce(df1_, model, field, '_pca');

    df2_ = pyal.restrict_to_interval(df2, epoch_fun=epoch) if epoch is not None else df
    df2_ = pyal.dim_reduce(df2_, model, field, '_pca');


    for targetIdx,target in enumerate(target_ids):
        df1__ = pyal.select_trials(df1_, df1_.target_id==target)
        df2__ = pyal.select_trials(df2_, df2_.target_id==target)

        all_id = df1__.reach_id.to_numpy()
        # to guarantee shuffled ids
        while ((all_id_sh := rng.permutation(all_id)) == all_id).all():
            continue
        all_id = all_id_sh

        # select the right number of trials to each target, use same subset ids for reaches to remain paired
        subset_ids = (df1__.reach_id.isin(all_id[:n_shared_trial]))
        df1__ = df1__[subset_ids]
        df2__ = df2__[subset_ids]

        for trial, (trial_rates1, trial_rates2) in enumerate(zip(df1__._pca, df2__._pca)):
            AllData1[0,targetIdx,trial, :, :] = trial_rates1
            AllData2[0,targetIdx,trial, :, :] = trial_rates2

    return AllData1, AllData2

def get_data_array(data_list: list[pd.DataFrame], epoch: Callable =None , area: str ='M1', model: Callable =None, n_components:int = 10) -> np.ndarray:
    """
    Applies the `model` to the `data_list` and return a data matrix of the shape: sessions x targets x trials x time x modes
    with the minimum number of trials and timepoints shared across all the datasets/targets.
    
    Parameters
    ----------
    `data_list`: list of pd.dataFrame datasets from pyalData (could also be a single dataset)
    `epoch`: an epoch function of the type `pyal.generate_epoch_fun()`
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
        rates_model = model.fit(rates)
        df_ = pyal.apply_dim_reduce_model(df_, rates_model, field, '_pca');

        for targetIdx,target in enumerate(target_ids):
            df__ = pyal.select_trials(df_, df_.target_id==target)
            all_id = df__.reach_id.to_numpy()
            # to guarantee shuffled ids
            while ((all_id_sh := rng.permutation(all_id)) == all_id).all():
                continue
            all_id = all_id_sh
            # select the right number of trials to each target
            df__ = pyal.select_trials(df__, lambda trial: trial.reach_id in all_id[:n_shared_trial])
            for trial, trial_rates in enumerate(df__._pca):
                AllData[session,targetIdx,trial, :, :] = trial_rates

    return AllData

# def custom_r2_func(y_true, y_pred):
#     "$R^2$ value as squared correlation coefficient, as per Gallego, NN 2020"
#     c = np.corrcoef(y_true.T, y_pred.T) ** 2
#     return np.diag(c[-int(c.shape[0]/2):,:int(c.shape[1]/2)])

# custom_r2_scorer = make_scorer(custom_r2_func)

# def get_data_array_and_vel(data_list: list[pd.DataFrame], epoch , area: str ='M1', n_components: int =10) -> np.ndarray:
#     """
#     Applies PCA to the data and return a data matrix of the shape: sessions x targets x  trials x time x PCs
#     with the minimum number of trials and timepoints shared across all the datasets/targets.
    
#     Parameters
#     ----------
#     `data_list`: list of pd.dataFrame datasets from pyal-data
#     `epoch`: an epoch function of the type `pyal.generate_epoch_fun`
#     `area`: area, either: 'M1', or 'S1', or 'PMd'

#     Returns
#     -------
#     `AllData`: np.array

#     Signature
#     -------
#     AllData = get_data_array(data_list, execution_epoch, area='M1', n_components=10)
#     all_data = np.reshape(AllData, (-1,10))
#     """
#     field = f'{area}_rates'
#     n_shared_trial = np.inf
#     for df in data_list:
#         for target in range(n_targets):
#             df_ = pyal.select_trials(df, df.target_id== target)
#             n_shared_trial = np.min((df_.shape[0], n_shared_trial))

#     n_shared_trial = int(n_shared_trial)

#     # finding the number of timepoints
#     df_ = pyal.restrict_to_interval(df_,epoch_fun=epoch)
#     n_timepoints = int(df_[field][0].shape[0])

#     # pre-allocating the data matrix
#     AllData = np.empty((len(data_list), n_targets, n_shared_trial, n_timepoints, n_components))
#     AllVel  = np.empty((len(data_list), n_targets, n_shared_trial, n_timepoints, 2))
#     for session, df in enumerate(data_list):
#         df_ = pyal.restrict_to_interval(df, epoch_fun=epoch)
#         rates = np.concatenate(df_[field].values, axis=0)
#         rates_model = PCA(n_components=n_components, svd_solver='full').fit(rates)
#         df_ = pyal.apply_dim_reduce_model(df_, rates_model, field, '_pca')
#         vel_mean = np.nanmean(pyal.concat_trials(df, 'pos'), axis=0)

#         for target in range(n_targets):
#             df__ = pyal.select_trials(df_, df_.target_id==target)
#             all_id = df__.trial_id.to_numpy()
#             rng.shuffle(all_id)
#             # select the right number of trials to each target
#             df__ = pyal.select_trials(df__, lambda trial: trial.trial_id in all_id[:n_shared_trial])
#             for trial, (trial_rates,trial_vel) in enumerate(zip(df__._pca, df__.pos)):
#                 AllData[session,target,trial, :, :] = trial_rates
#                 AllVel[session,target,trial, :, :] = trial_vel - vel_mean

#     return AllData, AllVel

# def time_trim(a,b):
#     l = min(a.shape[0],b.shape[0])
#     return a[:l],b[:l]

def _get_data_array(data_list: list[pd.DataFrame], epoch_L: int =None , area: str ='M1', model=None) -> np.ndarray:
    "Similar to `get_data_array` only returns an epoch of length `epoch_L` randomly chosen along each trial"
    if isinstance(data_list, pd.DataFrame):
        data_list = [data_list]
    if isinstance(model, int):
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
    n_timepoints = int(df_[field][0].shape[0])
    # n_timepoints = int(df_[field][0].shape[0])
    if epoch_L is None:
        epoch_L = n_timepoints
    else:
        assert epoch_L < n_timepoints, 'Epoch longer than data'
    
    # pre-allocating the data matrix
    AllData = np.zeros((len(data_list), len(target_ids), n_shared_trial, epoch_L, model.n_components))

    for session, df in enumerate(data_list):
        rates = np.concatenate(df[field].values, axis=0)
        rates_model = model.fit(rates)
        df_ = pyal.apply_dim_reduce_model(df, rates_model, field, '_pca');

        for targetIdx,target in enumerate(target_ids):
            df__ = pyal.select_trials(df_, df_.target_id==target)
            all_id = df__.reach_id.to_numpy()
            # to guarantee shuffled ids
            rng.shuffle(all_id)
            # select the right number of trials to each target
            df__ = pyal.select_trials(df__, lambda trial: trial.reach_id in all_id[:n_shared_trial])
            for trial, trial_rates in enumerate(df__._pca):
                time_idx = rng.integers(trial_rates.shape[0]-epoch_L)
                trial_data = trial_rates[time_idx:time_idx+epoch_L,:]
                AllData[session,targetIdx,trial, :, :] = trial_data

    return AllData

def trim_across_monkey_corr(paired_dfs):
    from scipy.stats import pearsonr
    across_corrs = {}
    #for each paired session
    all_across_corrs = []
    for _, df1__, df2__ in paired_dfs:

        df1 = pyal.restrict_to_interval(df1__, epoch_fun=exec_epoch)
        targets = np.unique(df1.target_id)
        df2 = pyal.restrict_to_interval(df2__, epoch_fun=exec_epoch)

        across_corrs = []
        #correlate pairs of reaches
        for pos1,pos2 in zip(df1.pos, df2.pos):
            # mse = np.mean((pos1-pos2)**2)
            # across_corrs.append(mse)

            r = [pearsonr(aa,bb)[0] for aa,bb in zip(pos1.T,pos2.T)]
            across_corrs.append(np.mean(np.abs(r)))
        all_across_corrs.append(across_corrs)

    return all_across_corrs