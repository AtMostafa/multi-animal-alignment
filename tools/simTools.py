from pickle import FALSE
from rnn.test import test_model
import os
import pyaldata as pyal
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import tools.dataTools as dt
import rnn.defs as rnn_defs
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter1d
import params


# PLOTTING FUNCTIONS
def get_colormap(categories, cmap = 'plasma_r', unique=False, truncate = True):
    if unique:
        color_labels = categories
    else:
        color_labels = np.unique(categories)
    if truncate:
        colors = [cm.get_cmap(cmap)(i) for i in np.linspace(0.1,0.9,len(color_labels))]
    else:
        colormap = cm.get_cmap(cmap, len(color_labels))
        colors = colormap(np.arange(0,len(color_labels)))
        
    color_map = dict(zip(color_labels, colors))
    return color_map

def get_data(datadir):
    return np.load(datadir+'.npy',allow_pickle = True).item()


def get_mse(seeds,sim):
    mses = []
    for seed in seeds:
        outdir = get_outdir(seed, sim)
        datadir, output, _ = test_model(outdir)
        datadir = rnn_defs.PROJ_DIR + datadir
        test_data = get_data(datadir)['test_set1']
        target = test_data['target']
        mse = ((target[:,50:,:] - output[:,50:,:])**2).mean(axis = 1)
        mse = mse.mean()
        mses.append(mse)
    return mses

def graph_sim_position(seed, sim_number, cmap = 'plasma', ax = None):
    outdir = get_outdir(seed, sim_number)
    datadir, output, activity1 = test_model(outdir)
    data = get_data(rnn_defs.PROJ_DIR + datadir)
    params = data['params']
    target_info = np.mod(data['test_set1']['target_id']-3,8)

    use_velocities = params['use_velocities']

    if use_velocities:
        dt = params['dt']
        positions = np.zeros(output.shape)
        for j in range(output.shape[1]):
            positions[:,j,:] = positions[:,j-1,:] + output[:,j,:]*dt
        graph_position(positions, target_info, cmap = cmap, ax = ax)
    else:
        graph_position(output, target_info, cmap = cmap, ax = ax)

def graph_position (positions, task_info = None, cmap= None, ax = None, **plot_kwargs):

    ax = ax or plt.gca()

    # make color map based on different stimuli
    if task_info is not None:
        colormap = get_colormap(task_info, cmap = cmap) 
    else:
        colormap = None
        
    y_pos = rnn_defs.MAX_Y_POS

    # graph positions
    for i in range(positions.shape[0]):
        if colormap is not None:
            ax.plot(positions[i,:,0],positions[i,:,1], c = colormap[task_info[i]],
                    linestyle = '-', linewidth = 1, marker = None, **plot_kwargs)
        else:
            ax.plot(positions[i,:,0],positions[i,:,1], c = 'k',
                    linestyle = '-', linewidth = 1, marker = None, **plot_kwargs)
        ax.set_aspect(1)
        ax.set_xlim([-y_pos-2,y_pos+2])
        ax.set_ylim([-y_pos-2,y_pos+2])
    ax.set_axis_off()

def get_outdir(seed, sim_number):
    outdir = rnn_defs.PROJ_DIR + rnn_defs.RESULTS_FOLDER + '/' + str(seed) + '/' + str(sim_number) + '/' 
    return outdir

def get_processed_pyaldata(seed, sim_number, epoch_fun = None, calc_kinematics = True, 
        subtract_mean = False):
    pyal_df = model_to_pyaldata(seed, sim_number, calc_kinematics)
    pyal_df = pyal.add_movement_onset(pyal_df, method = 'threshold', s_thresh = 9)
    pyal_df = pyal.smooth_signals(pyal_df, ["MCx_rates"])

    if epoch_fun is not None:
        pyal_df = pyal.restrict_to_interval(pyal_df, epoch_fun = epoch_fun)

    if subtract_mean: 
        pyal_df = pyal.subtract_cross_condition_mean(pyal_df)  

    return pyal_df

# convert current model results to pyaldata
def model_to_pyaldata (seed, sim_number, calc_kinematics = True):
    """
    Converts model results to Pyaldata format.

    Parameters
    ----------
    outdir: str
        where model is saved

    Returns
    -------
    df: Pandas dataframe
        in pyaldata format

    """
    #get directories and model output
    outdir = get_outdir(seed, sim_number)
    datadir, output, activity1 = test_model(outdir)

    #get data
    datname = rnn_defs.PROJ_DIR + datadir
    data = np.load(datname+'.npy',allow_pickle = True).item()
    test_data = data['test_set1']
    params = data['params']

    # columns needed for pyaldata
    column_names = ['seed', 'sim_number', 'target_id', 'target_param', 'trial_id', 'bin_size', 
        'idx_trial_start', 'idx_target_on', 'idx_go_cue', 'idx_trial_end', 
        'MCx_rates']
    df = pd.DataFrame(columns = column_names)

    ntrials = len(test_data['idxoutofall'])
    tsteps = params['tsteps']
    dt = params['dt']
    #populate columns
    df['target_id'] = test_data['target_id']
    df['seed'] = seed
    df['sim_number'] = sim_number
    df['target_param'] = test_data['target_param']
    df['trial_id'] = test_data['idxoutofall']
    df['bin_size'] = params['dt']
    df['idx_trial_start'] = 0
    df['idx_target_on'] = test_data['cue_onset']
    df['idx_go_cue'] = test_data['go_onset']
    df['idx_trial_end'] = tsteps-1
    df['MCx_rates'] =[activity1[i,:] for i in range(ntrials)] 
    df['pos'] = [output[i,:] for i in range(ntrials)] 

    if calc_kinematics:
        #calculate vel, accel, and speed
        vels = np.zeros((ntrials, tsteps, 2))
        # accels = np.zeros((ntrials, tsteps, 2))
        speed = np.zeros((ntrials, tsteps, 2))
        pos = output
        for trial in range(ntrials):
            go_step = df['idx_go_cue'][trial]
            for tstep in range(go_step, tsteps):
                #calculate velocity
                vels[trial, tstep, 0] = (pos[trial,tstep,0]-pos[trial,tstep-1,0])/dt
                vels[trial, tstep, 1] = (pos[trial,tstep,1]-pos[trial,tstep-1,1])/dt
            
        df['vel'] = [vels[i] for i in range(ntrials)]
    return df
 
def get_cc_within(dfs, n_components, epoch_fun = None):
    data = dt.get_data_array(dfs, epoch_fun, area = 'MCx', model = n_components)

    n_shared_trial1 = data.shape[2]
    trialList1 = np.arange(n_shared_trial1)
    ccs=[]
    for session, sessionData in enumerate(data):
        r = []
        for n in range(params.n_iter*10):
            params.rng.shuffle(trialList1)
            # non-overlapping randomised trials
            trial1 = trialList1[:n_shared_trial1//2]
            trial2 = trialList1[-(n_shared_trial1//2):]
            data1 = np.reshape(sessionData[:,trial1,:,:], (-1,n_components))
            data2 = np.reshape(sessionData[:,trial2,:,:], (-1,n_components))
            r.append(dt.canoncorr(data1, data2))
        ccs.append(r)
    ccs = np.array(ccs)
    ccs = np.percentile(ccs, 99, axis=1).T
    return ccs

def get_cc_across(dfs, n_components, epoch_fun = None):

    pairFileList1 = []
    for I in range(len(dfs)):
        for J in range(len(dfs)):
            if J<=I: continue  # to repetitions
            pairFileList1.append((I,J))

    side1df = [dfs[i] for i,_ in pairFileList1]
    side2df = [dfs[j] for _,j in pairFileList1]
    AllData1 = dt.get_data_array(side1df, epoch_fun, area='MCx', model=n_components)
    AllData2 = dt.get_data_array(side2df, epoch_fun, area='MCx', model=n_components)
    
    _,_, min_trials, min_time,_ = np.min((AllData1.shape,AllData2.shape),axis=0)
    ccs=[]
    for sessionData1,sessionData2 in zip(AllData1,AllData2):
        data1 = np.reshape(sessionData1[:,:min_trials,:min_time,:], (-1,n_components))
        data2 = np.reshape(sessionData2[:,:min_trials,:min_time,:], (-1,n_components))
        ccs.append(dt.canoncorr(data1, data2))
    
    ccs = np.array(ccs).T

    return ccs

def get_cc_across_sims(dfs1, dfs2, n_components, epoch_fun = None):

    pairFileList1 = []
    for I in range(len(dfs1)):
        for J in range(len(dfs2)):
            if J<=I: continue  # to repetitions
            pairFileList1.append((I,J))

    side1df = [dfs1[i] for i,_ in pairFileList1]
    side2df = [dfs2[j] for _,j in pairFileList1]
    AllData1 = dt.get_data_array(side1df, epoch_fun, area='MCx', model=n_components)
    AllData2 = dt.get_data_array(side2df, epoch_fun, area='MCx', model=n_components)
    
    _,_, min_trials, min_time,_ = np.min((AllData1.shape,AllData2.shape),axis=0)
    ccs=[]
    for sessionData1,sessionData2 in zip(AllData1,AllData2):
        data1 = np.reshape(sessionData1[:,:min_trials,:min_time,:], (-1,n_components))
        data2 = np.reshape(sessionData2[:,:min_trials,:min_time,:], (-1,n_components))
        ccs.append(dt.canoncorr(data1, data2))
    
    ccs = np.array(ccs).T

    return ccs

def trim_across_rnn_corr(allDF1:list[pd.DataFrame],allDF2:list[pd.DataFrame]):
    across_corrs = {}
    for dfi, df1__ in enumerate(allDF1):
        df1 = pyal.restrict_to_interval(df1__, epoch_fun=rnn_defs.exec_epoch)
        targets = np.unique(df1.target_id)
        across_corrs[df1.seed[0]]={}
        for dfj, df2__ in enumerate(allDF2):
            df2 = pyal.restrict_to_interval(df2__, epoch_fun=rnn_defs.exec_epoch)
            across_corrs[df2.seed[0]] = {} if df2.seed[0] not in across_corrs.keys() else across_corrs[df2.seed[0]]
            across_corrs[df1.seed[0]][df2.seed[0]]=[]
            for target in targets:
                df1_ = pyal.select_trials(df1, df1.target_id == target)
                df2_ = pyal.select_trials(df2, df2.target_id == target)
                for i, pos1 in enumerate(df1_.pos):
                    for j, pos2 in enumerate(df2_.pos):
                        r = [pearsonr(aa,bb)[0] for aa,bb in zip(pos1.T,pos2.T)]
                        across_corrs[df1_.seed[0]][df2_.seed[0]].append(np.mean(np.abs(r)))

        # make the across correlations symmetrical!
        for  df2_session, val in across_corrs[df1__.seed[0]].items():
            across_corrs[df2_session][df1__.seed[0]] = val

    return across_corrs

def trim_across_seeds_rnn_corr(allDF1:list[pd.DataFrame]):
    across_corrs = {}
    for dfi, df1__ in enumerate(allDF1):
        df1 = pyal.restrict_to_interval(df1__, epoch_fun=rnn_defs.exec_epoch)
        targets = np.unique(df1.target_id)
        across_corrs[df1.seed[0]]={}
        for dfj, df2__ in enumerate(allDF1):
            df2 = pyal.restrict_to_interval(df2__, epoch_fun=rnn_defs.exec_epoch)
            across_corrs[df2.seed[0]] = {} if df2.seed[0] not in across_corrs.keys() else across_corrs[df2.seed[0]]
            if dfj <= dfi: continue
            across_corrs[df1.seed[0]][df2.seed[0]]=[]
            for target in targets:
                df1_ = pyal.select_trials(df1, df1.target_id == target)
                df2_ = pyal.select_trials(df2, df2.target_id == target)
                for i, pos1 in enumerate(df1_.pos):
                    for j, pos2 in enumerate(df2_.pos):
                        r = [pearsonr(aa,bb)[0] for aa,bb in zip(pos1.T,pos2.T)]
                        across_corrs[df1_.seed[0]][df2_.seed[0]].append(np.mean(np.abs(r)))
    return across_corrs


def trim_within_rnn_corr(allDF1:list[pd.DataFrame]):
    within_corrs = {}
    for dfi, df1__ in enumerate(allDF1):
        df1 = pyal.restrict_to_interval(df1__, epoch_fun=rnn_defs.exec_epoch)
        targets = np.unique(df1.target_id)
        within_corrs[df1.seed[0]]=[]
        for target in targets:
            df1_ = pyal.select_trials(df1, df1.target_id == target)
            for n in range(10):
                shuffled = df1_.sample(frac=1)
                result = np.array_split(shuffled, 2) 
                for i, pos1 in enumerate(result[0].pos):
                    for j, pos2 in enumerate(result[1].pos):
                        r = [pearsonr(aa,bb)[0] for aa,bb in zip(pos1.T,pos2.T)]
                        within_corrs[df1_.seed[0]].append(np.mean(np.abs(r)))

    return within_corrs
