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

# def subsample_neurons(df, n_sample = None, p = None):
#     assert sum([arg is not None for arg in [n_sample, p]]) == 1
#     fields = [col for col in df.columns.values
#                 if col.endswith("spikes") or col.endswith("rates")]
#     n_neurons = df[fields[0]].values[0].shape[1]

#     # choose random sample by number of neurons or percentage of population
#     if n_sample is not None:
#         idx = np.random.choice(n_neurons, n_sample)
#     else:
#         idx = np.random.choice(n_neurons, int(n_neurons*p))
        
#     for field in fields:
#         df[field] = [trial[field][:,idx] for (i, trial) in df.iterrows()]
    
#     return df


def get_processed_pyaldata(seed, sim_number, epoch_fun = None, calc_kinematics = True, 
        subtract_mean = True):
    pyal_df = model_to_pyaldata(seed, sim_number, calc_kinematics)
    pyal_df = pyal.add_movement_onset(pyal_df, method = 'threshold', s_thresh = 9)
    pyal_df = pyal.smooth_signals(pyal_df, ["MCx_rates"])

    if epoch_fun is not None:
        pyal_df = pyal.restrict_to_interval(pyal_df, epoch_fun = epoch_fun)

    if subtract_mean: 
        pyal_df = pyal.subtract_cross_condition_mean(pyal_df)     
    return pyal_df

# def get_training_data(outdir):
#     train_data = np.load(outdir+'training.npy',allow_pickle = True).item()
#     return train_data

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
    
# def get_pyaldata (activity1, test_data, dt):
#     """
#     Converts model results to Pyaldata format.

#     Parameters
#     ----------
#     outdir: str
#         where model is saved

#     Returns
#     -------
#     df: Pandas dataframe
#         in pyaldata format

#     """
#     # columns needed for pyaldata
#     column_names = ['bin_size', 'idx_target_on', 'idx_go_cue', 'MCx_rates']
#     df = pd.DataFrame(columns = column_names)

#     #populate columns
#     df['idx_target_on'] = test_data.data['cue_onset']
#     df['idx_go_cue'] = test_data.data['go_onset']
#     df['MCx_rates'] =[activity1[i,:] for i in range(activity1.shape[0])] 
#     df['bin_size'] = dt

#     return df
    

# def save_pcas(pcas, sim_number, sample_trials=None):
#     """
#     Save pcas as files
    
#     Parameters
#     ----------
#     pcas: array
#         m items x n samples x p PCs
#     """
#     params = '_'.join(filter(None, [str(sim_number), 
#                                     str(sample_trials) if sample_trials is not None else None]))
#     np.save(savdir+'pcas_'+params, pcas)

# def get_pcas_path(sim_number, group, start_point = None, rel_start = 0, rel_end = 0, n_sample = None, p =None, sample_trials=None):
#     params = '_'.join(filter(None, [group, str(sim_number),  start_point, str(rel_start), str(rel_end),
#                                     str(n_sample) if n_sample is not None else None, 
#                                     str(p) if p is not None else None,
#                                     str(sample_trials) if sample_trials is not None else None]))
#     return savdir+'pcas_'+params+'.npy'

# def sample_pca(seed, sim_number, n_components, ntrials = 64, epoch_fun = None):
#     df = get_processed_pyaldata(seed, sim_number, epoch_fun = epoch_fun, subtract_mean=False)
#     df = pyal.dim_reduce(df, PCA(n_components), 'MCx_rates', 'both_pca')
#     # df = pyal.subtract_cross_condition_mean(df)
#     df = df.groupby('target_id').head(ntrials/rnn_defs.n_targets)
#     pca = np.concatenate(df['both_pca'].values, axis=0)
#     return pca

# def sample_pcas(seeds, sim_number, group, ntrials = 128, start_point = start_point, rel_start = rel_start, rel_end = rel_end, n_sample =None, p = None):
#     pcas = []
#     for seed in seeds: 
#         pca = sample_pca(seed, sim_number, group, ntrials, start_point, rel_start, rel_end, n_sample, p)
#         pcas.append(pca)
#     save_pcas(pcas, sim_number, group, start_point, rel_start, rel_end, n_sample, p, sample_trials = ntrials)
#     return pcas


def get_pca(seed, sim_number, n_components, epoch_fun = None):
    df = get_processed_pyaldata(seed, sim_number, epoch_fun = epoch_fun)
    df = pyal.dim_reduce(df, PCA(n_components), 'MCx_rates', 'both_pca')
    pca = np.concatenate(df['both_pca'].values, axis=0)
    return pca

def get_pcas(seeds, sim_number, n_components, epoch_fun = None, replace = False):
    # path = get_pcas_path(sim_number, group, start_point, rel_start, rel_end, n_sample, p)
    # if os.path.exists(path) and not replace:
    #     pcas = np.load(path,allow_pickle = True)
    #     # print(path)

    # else:
    pcas = []
    for seed in seeds: 
        pca = get_pca(seed, sim_number, n_components, epoch_fun = epoch_fun)
        pcas.append(pca)
    # save_pcas(pcas, sim_number, group, start_point, rel_start, rel_end, n_sample, p)
    return pcas

def get_outputs(seeds, sim_number, epoch_fun = None):
    outputs = []
    for seed in seeds:
        df = get_processed_pyaldata(seed, sim_number, epoch_fun = epoch_fun)
        output = np.concatenate(df['pos'].values, axis = 0)
        outputs.append(output)
    return outputs

def get_cc_within_seeds (seeds, sim_number, n_components, epoch_fun = None):
    pcas = get_pcas(seeds, sim_number, n_components, epoch_fun = epoch_fun)

    ccs = []
    for i, pca1 in enumerate(pcas):
        for j, pca2 in enumerate(pcas[i+1:]):
            n = min(pca1.shape[0],pca2.shape[0])
            ccs.append(dt.canoncorr(pca1[:n,:],pca2[:n,:]))
    ccs = np.array(ccs).T

    return ccs

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
 

# def get_success_rate (seeds, sim_number, group):
#     n_success = 0
#     for seed in seeds:
#         outdir = st.get_outdir(seed, sim_number, group)
#         finished_training = st.get_training_data(outdir)['finished_training']
#         if finished_training:
#             n_success +=1
    
#     return (n_success/len(seeds))


def get_cc_between_sims (seeds1, seeds2, sim1, sim2, n_components, epoch_fun = None):
    sim1_pcas = get_pcas(seeds1, sim1, n_components, epoch_fun)
    sim2_pcas = get_pcas(seeds2, sim2, n_components, epoch_fun)

    ccs = []
    for pca1 in (sim1_pcas):
        for pca2 in (sim2_pcas):
            ccs.append(dt.canoncorr(pca1,pca2))
    ccs = np.array(ccs).T

    return ccs

# def plot_ccs(ccs, title, ax = None, add_labels = True, color = 'b'):
#     if ax is None:
#         _, ax = plt.subplots(dpi=100, figsize = (4,3))
#     utility.shaded_errorbar(ax, ccs, color=color, marker = 'o')
#     ax.set_title(title)
#     if add_labels:
#         ax.set_xlabel('components')
#         ax.set_ylabel('canonical correlation')  

# def plot_ccs_upperbounds(across_ccs, within_ccs_1, within_ccs_2, label1, label2, ax = None):
#     if ax is None:
#         _,ax = plt.subplots(dpi=100, figsize = (4,3))
#     utility.shaded_errorbar(ax, across_ccs, color='b', marker = 'o', label = label1 + ' vs ' + label2)
#     utility.shaded_errorbar(ax, within_ccs_1, color='g', marker = 'o', label = label1)
#     utility.shaded_errorbar(ax, within_ccs_2, color='r', marker = 'o', label = label2)
#     ax.set_title(label1 + ' vs ' + label2)
#     ax.legend()
#     ax.set_xlabel('components')
#     ax.set_ylabel('canonical correlation') 

# def successful_sims(sim_numbers, seeds, group):
#     successful_sims = []
#     for sim_number in sim_numbers:
#         try:
#             success_rate = get_success_rate(seeds, sim_number, group)
#             if success_rate > 0.5:
#                 successful_sims.append(sim_number)
#             else:
#                 title = sim_label[sim_number]
#                 print(sim_number, title, 'failed:', success_rate)
#         except:
#             title = sim_label[sim_number]
#             print(group, 'failed to get', sim_number, title)

#     return successful_sims

# def graph_cc_comparison(seeds1, seeds2, sim1, sim2, group1, group2, pca_dims, start_point, rel_start, rel_end, n_sample = None, p = None):
#     ccs_1 = get_cc_within_seeds(seeds1, sim1, group1, pca_dims, start_point, rel_start, rel_end, n_sample, p)
#     ccs_2 = get_cc_within_seeds(seeds2, sim2, group2, pca_dims, start_point, rel_start, rel_end, n_sample, p)
#     ccs = get_cc_between_sims(seeds1, seeds2, sim1, sim2, pca_dims, group1, group2, start_point, rel_start, rel_end, n_sample,p)

#     fig,ax = plt.subplots(ncols=1, figsize=(3,2))
#     utility.shaded_errorbar(ax, np.arange(1,pca_dims+1), ccs, color='indigo', marker = 'o', label=f'Across')
#     utility.shaded_errorbar(ax, np.arange(1,pca_dims+1), ccs_1, color='cornflowerblue', marker = '<', ls='--', label=f'Within networks 1')
#     utility.shaded_errorbar(ax, np.arange(1,pca_dims+1), ccs_2, color='palevioletred', marker = '>', ls=':', label=f'Within networks 2')

#     ax.set_ylim([0,1.05])
#     ax.set_xlim([.5,pca_dims+.5])
#     ax.set_xlabel('Neural mode')
#     # ax.set_title(f'{defs.areas[2]} Alignment')
#     ax.legend(loc=(0.01,0.05))
#     ax.set_ylabel('Canonical correlation')
#     fig.tight_layout()

#     params = '_'.join(filter(None, [group1, sim_label[sim1], group2, sim_label[sim2], 
#                                         start_point, str(rel_start), str(rel_end),
#                                         str(n_sample) if n_sample is not None else None, 
#                                         str(p) if p is not None else None]))
#     fig.savefig(figdir + 'compare_' + params + '.pdf', format='pdf', bbox_inches='tight')

def get_cc_behav_corr_between_seeds(seeds, sim_number, n_components, epoch_fun, redo=False):
    path = rnn_defs.PROJ_DIR + rnn_defs.RESULTS_FOLDER + "corr_within_"+ str(sim_number)+'.npy'
    if os.path.exists(path) and not redo:
        corr_within = np.load(path,allow_pickle = True)
    else:
        corr_within =[]

        dfs = []
        for seed in seeds:
            df = get_processed_pyaldata(seed,sim_number, epoch_fun, calc_kinematics=True)
            df = pyal.dim_reduce(df, PCA(n_components), 'MCx_rates', 'both_pca')
            dfs.append(df)

        for i,seed1 in enumerate(seeds):
            #info from first network
            df1 = dfs[i]
            pca1 = np.concatenate(df1['both_pca'].values, axis = 0)
            targets = np.unique(df1.target_id)

            for j,seed2 in enumerate(seeds):
                if i <=j: continue
                #info from second network
                df2 = dfs[j]
                pca2 = np.concatenate(df2['both_pca'].values, axis = 0)

                #look at behavioral correlation for matched targets
                behav_corr = []
                for target in targets:
                    df1_ = pyal.select_trials(df1, df1.target_id == target)
                    df2_ = pyal.select_trials(df2, df2.target_id == target)

                    #for each trial
                    for i, pos1 in enumerate(df1_.pos):
                        for j, pos2 in enumerate(df2_.pos):
                            r = [pearsonr(aa,bb)[0] for aa,bb in zip(pos1.T,pos2.T)]
                            behav_corr.append(np.mean(np.abs(r)))
                
                #get CCA            
                corr_within.append((dt.canoncorr(pca1, pca2)[:4].mean() , np.mean(behav_corr)))
        np.save(path, corr_within)
    return corr_within

def get_cc_behav_corr_across_sims(seeds1, seeds2, sim_number1, sim_number2, n_components, epoch_fun, redo = False):
    path = rnn_defs.PROJ_DIR + rnn_defs.RESULTS_FOLDER + "corr_across_"+ str(sim_number1)+ '_' + str(sim_number2)+'.npy'
    if os.path.exists(path) and not redo:
        corr_across = np.load(path,allow_pickle = True)
    
    else:
        corr_across =[]

        #get data
        dfs1 = []
        for seed in seeds1:
            df = get_processed_pyaldata(seed,sim_number1, epoch_fun, calc_kinematics=True)
            df = pyal.dim_reduce(df, PCA(n_components), 'MCx_rates', 'both_pca')
            dfs1.append(df)
        
        dfs2 = []
        for seed in seeds2:
            df = get_processed_pyaldata(seed,sim_number2, epoch_fun, calc_kinematics=True)
            df = pyal.dim_reduce(df, PCA(n_components), 'MCx_rates', 'both_pca')
            dfs2.append(df)

        #calculate correlations and CCs
        for i,df1 in enumerate(dfs1):
            #info from network in first sim
            pca1 = np.concatenate(df1['both_pca'].values, axis = 0)
            targets = np.unique(df1.target_id)

            for j,df2 in enumerate(dfs2):
                #info from network in second sim
                pca2 = np.concatenate(df2['both_pca'].values, axis = 0)

                #look at behavioral correlation for matched targets
                behav_corr = []
                for target in targets:
                    df1_ = pyal.select_trials(df1, df1.target_id == target)
                    df2_ = pyal.select_trials(df2, df2.target_id == target)

                    #for each trial
                    for m, pos1 in enumerate(df1_.pos):
                        for n, pos2 in enumerate(df2_.pos):
                            r = [pearsonr(aa,bb)[0] for aa,bb in zip(pos1.T,pos2.T)]
                            behav_corr.append(np.mean(np.abs(r)))

                #get CCA            
                corr_across.append((dt.canoncorr(pca1, pca2)[:4].mean() , np.mean(behav_corr)))
        np.save(path, corr_across)

    return corr_across