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

raster_example = ('Chewie_CO_FF_2016-10-13.p', 'Mihili_CO_VR_2014-03-03.p')
MAX_HISTORY = 3  #int: no of bins to be added as history
BIN_SIZE = .03  # sec
WINDOW_prep = (-.4, .05)  # sec
WINDOW_exec = (-.05, .40)  # sec
n_components = 10  # min between M1 and PMd
areas = ('M1', 'PMd', 'MCx')
n_targets = 8
output_dims = 2

prep_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on',
                                     rel_start=int(WINDOW_prep[0]/BIN_SIZE),
                                     rel_end=int(WINDOW_prep[1]/BIN_SIZE)
                                    )
exec_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on',
                                     rel_start=int(WINDOW_exec[0]/BIN_SIZE),
                                     rel_end=int(WINDOW_exec[1]/BIN_SIZE)
                                    )
fixation_epoch = pyal.generate_epoch_fun(start_point_name='idx_target_on',
                                         rel_start=int(WINDOW_prep[0]/BIN_SIZE),
                                         rel_end=int(WINDOW_prep[1]/BIN_SIZE)
                                        )
exec_epoch_decode = pyal.generate_epoch_fun(start_point_name='idx_movement_on',
                                     rel_start=int(WINDOW_exec[0]/BIN_SIZE) - MAX_HISTORY,
                                     rel_end=int(WINDOW_exec[1]/BIN_SIZE)
                                    )

def get_target_id(trial):
    return int(np.round((trial.target_direction + np.pi) / (0.25*np.pi))) - 1

def prep_general (df):
    "preprocessing general!"
    time_signals = [signal for signal in pyal.get_time_varying_fields(df) if 'spikes' in signal]
    df["target_id"] = df.apply(get_target_id, axis=1)  # add a field `target_id` with int values
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
    
    
    return df_

def custom_r2_func(y_true, y_pred):
    "$R^2$ value as squared correlation coefficient, as per Gallego, NN 2020"
    c = np.corrcoef(y_true.T, y_pred.T) ** 2
    return np.diag(c[-int(c.shape[0]/2):,:int(c.shape[1]/2)])

custom_r2_scorer = make_scorer(custom_r2_func)

def get_data_array_and_pos(data_list: list[pd.DataFrame], epoch , area: str ='M1', n_components: int =10, normalize_pos = False) -> np.ndarray:
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
    def normal_mov(df: pd.DataFrame, field:str ='hTrjB') -> pd.DataFrame:
        """
        normalises based on 99th percentile for the magnitude of the movement
        """
        df = df.copy()
        magnitude = np.percentile(np.abs(np.concatenate(df[field]).flatten()), 99)
        df[field] = [pos/magnitude for pos in df[field]]
        return df

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
    AllVel  = np.empty((len(data_list), n_targets, n_shared_trial, n_timepoints, 2))
    for session, df in enumerate(data_list):
        df_ = pyal.restrict_to_interval(df, epoch_fun=epoch)
        pos_mean = np.nanmean(pyal.concat_trials(df, 'pos'), axis=0)
        df_.pos = [pos - pos_mean for pos in df_.pos] #TODO: check if this is correct
        if normalize_pos:
            df_ = normal_mov(df_,'pos')

        rates = np.concatenate(df_[field].values, axis=0)
        rates_model = PCA(n_components=n_components, svd_solver='full').fit(rates)
        df_ = pyal.apply_dim_reduce_model(df_, rates_model, field, '_pca')
        
        for target in range(n_targets):
            df__ = pyal.select_trials(df_, df_.target_id==target)
            all_id = df__.trial_id.to_numpy()
            rng.shuffle(all_id)
            # select the right number of trials to each target
            df__ = pyal.select_trials(df__, lambda trial: trial.trial_id in all_id[:n_shared_trial])
            for trial, (trial_rates,trial_vel) in enumerate(zip(df__._pca, df__.pos)):
                AllData[session,target,trial, :, :] = trial_rates
                AllVel[session,target,trial, :, :] = trial_vel

    return AllData, AllVel
