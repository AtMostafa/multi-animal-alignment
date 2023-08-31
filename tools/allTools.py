#TODO: take care of correct loading dir for params
import params
import os
import numpy as np
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib as mpl
import pyaldata as pyal
from scipy.stats import wilcoxon
import pandas as pd
from sklearn.decomposition import PCA
import pickle
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import KFold, cross_val_score
from tools import lstm
from sklearn.feature_selection import r_regression
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tools import dataTools as dt
from tools import utilityTools as utility
# from tools.monkey_data_selection import GoodDataList
monkey_defs = params.monkey_defs
mouse_defs = params.mouse_defs
random_walk_defs = params.random_walk_defs

rng = np.random.default_rng(np.random.SeedSequence(12345))

####################################################
# Get data
####################################################
## monkey
def get_full_monkey_data(data_list):
    full_list_MCx = []
    for animal, sessionList in data_list[monkey_defs.areas[2]].items():
        if 'Mr' in animal:
            continue  # to remove MrT
        full_list_MCx.append((animal,sessionList))
    full_list_MCx = [(animal,session) for animal,sessions in full_list_MCx for session in set(sessions)]
    # load the DFs
    allDFs_MCx = []
    for animal, session in full_list_MCx:
        path = params.root/animal/session
        allDFs_MCx.append(monkey_defs.prep_general(dt.load_pyal_data(path)))

    return full_list_MCx, allDFs_MCx

def get_example_monkey_data(epoch = None):
    raster_example = monkey_defs.raster_example
    raster_example_df = []
    for session in raster_example:
        path = params.root/session.split('_')[0]/session
        df = monkey_defs.prep_general(dt.load_pyal_data(path))
        if epoch is not None:
            df = pyal.restrict_to_interval(df, epoch_fun=epoch)
        raster_example_df.append(df)
    return raster_example_df

## mouse
def get_full_mouse_data(prep_pull = False):
    defs = mouse_defs
    
    animalList = ['mouse-data']
    animalFiles = []
    for animal in animalList:
        animalFiles.extend(utility.find_file(params.root / animal, 'mat'))

    AllDFs=[]
    for fname in animalFiles:
        df = dt.load_pyal_data(fname)
        df['mouse'] = fname.split(os.sep)[-1][fname.split(os.sep)[-1].find('WR'):].split('_')[0]
        df['file'] = fname.split(os.sep)[-1]
        df['session']=df['file']
        if prep_pull:
            df = defs.prep_pull_mouse(df)
        else:
            df = defs.prep_general_mouse(df)
        df['pos']=df['hTrjB']
        AllDFs.append(df)

    allDFs_M1 = []
    for df in AllDFs:
        if 'M1_rates' in df.columns:
            allDFs_M1.append(df)

    allDFs_Str = []
    for df in AllDFs:
        if 'Str_rates' in df.columns:
            allDFs_Str.append(df)
            
    return allDFs_M1, allDFs_Str

def get_example_mouse_data():
    example = mouse_defs._example
    animal = 'mouse-data'
    
    example_df = []
    for session in example:
        path = params.root / animal / session
        df = dt.load_pyal_data(path)
        path = str(path)
        df['mouse'] = path.split(os.sep)[-1][path.split(os.sep)[-1].find('WR'):].split('_')[0]
        df['file'] = path.split(os.sep)[-1]
        df['session']=df['file']
        df = mouse_defs.prep_general_mouse(df)
        df['pos']=df['hTrjB']
        df = pyal.restrict_to_interval(df, epoch_fun=mouse_defs.exec_epoch)
        example_df.append(df)
        
    return example_df

## monkey rw
### Get data
def get_full_random_walk_data(GoodDataList_RW):
    defs = random_walk_defs

    full_list_MCx = []
    for animal, sessionList in GoodDataList_RW['MCx'].items():
        full_list_MCx.append((animal,sessionList))
    full_list_MCx = [(animal,session) for animal,sessions in full_list_MCx for session in set(sessions)]
    # load the DFs
    allDFs_MCx = []
    allDFs_exec_MCx = []
    for animal, session in full_list_MCx:
        path = params.root/'random_walk'/animal/session
        df_ = defs.prep_general(dt.load_pyal_data(path))

        #separate into reaches
        df_ = defs.get_reaches_df(df_)
        df_['reach_id'] = range(len(df_))

        #subset center-out trials
        df_ = df_[df_.center_dist < defs.subset_radius]
        df_ = df_.reset_index()

        #execution epoch
        for col in df_.columns:  #TODO: placeholder to prevent bug in pyaldata
            if 'unit_guide' in col:
                df_ = df_.drop([col], axis = 1)
        df_ = pyal.add_movement_onset(df_)
        allDFs_MCx.append(df_)

        df_ = pyal.restrict_to_interval(df_, epoch_fun=defs.exec_epoch)
        allDFs_exec_MCx.append(df_)
    
        
    return full_list_MCx, allDFs_MCx, allDFs_exec_MCx

def get_paired_dfs(GoodDataList_RW, MCx_list, allDFs_MCx, allDFs_exec_MCx):
    defs = random_walk_defs

    ref_file = 'Chewie_RT_CS_2016-10-21.mat' #TODO: put in defs
    ref_i = [y for x,y in MCx_list].index(ref_file)
    df1 = allDFs_exec_MCx[ref_i]

    Mihili_files = GoodDataList_RW['MCx']['Mihili']
    MrT_files = GoodDataList_RW['MCx']['MrT']
    comparison_files = Mihili_files + MrT_files

    paired_dfs = []
    for ex_file in comparison_files:
        
        ex_i = [y for x,y in MCx_list].index(ex_file)
        df2 = allDFs_exec_MCx[ex_i]

        #subset dataframes with matched reaches
        df1_idx, df2_idx = defs.get_matched_reaches_idx(df1, df2)
        df1_subset = df1.iloc[df1_idx]
        df2_subset = df2.iloc[df2_idx]

        #get dataframes from whole-trial data
        df1_ = pd.DataFrame({'reach_id':df1_subset.reach_id}).merge(allDFs_MCx[ref_i])
        df2_ = pd.DataFrame({'reach_id':df2_subset.reach_id}).merge(allDFs_MCx[ex_i])

        #set target ids
        print(ex_file, len(df1_) - (df1_.target_group.values == df2_.target_group.values).sum(), 'diff target groups')
        df1_.target_group = df2_.target_group.values 
        df1_['target_id'] = df1_.target_group.values
        df2_['target_id'] = df2_.target_group.values

        #only keep target groups with enough trials
        counter = Counter(df1_.target_group)
        subset_target_groups = [k for k, c in counter.items() if c >= defs.min_trials_per_target]
        df1_ = df1_[df1_.target_group.isin(subset_target_groups)]
        df2_ = df2_[df2_.target_group.isin(subset_target_groups)]
        
        print(len(subset_target_groups), 'target groups left')

        paired_dfs.append((ex_file, df1_, df2_))
    
    return paired_dfs

## general
def get_paired_files_monkey(allDFs):
    pairFileList = []
    for I, df1 in enumerate(allDFs):
        for J, df2 in enumerate(allDFs):
            animal1 = df1.monkey[0]
            animal2 = df2.monkey[0]
            if J<=I or animal1 == animal2: continue  # to repetitions
            if 'Chewie' in animal1 and 'Chewie' in animal2: continue 
            pairFileList.append((I,J))
    return pairFileList

def get_paired_files_mouse(allDFs):
    pairFileList = []
    for I, df1 in enumerate(allDFs):
        for J, (df2) in enumerate(allDFs):
            if J<=I or df1.mouse[0] == df2.mouse[0]: continue  # repetitions
            pairFileList.append((I,J))
    return pairFileList


####################################################
# CCA plots 
####################################################
def get_ccs(side1df, side2df, epoch, area, n_components, use_procrustes = False): 

    #get data  
    AllData1 = dt.get_data_array(side1df, epoch, area=area, model=n_components)
    AllData2 = dt.get_data_array(side2df, epoch, area=area, model=n_components)
    _,_, min_trials, min_time,_ = np.min((AllData1.shape,AllData2.shape),axis=0)

    #get ccs
    ccs=[]
    for sessionData1,sessionData2 in zip(AllData1,AllData2):
        data1 = np.reshape(sessionData1[:,:min_trials,:min_time,:], (-1,n_components))
        data2 = np.reshape(sessionData2[:,:min_trials,:min_time,:], (-1,n_components))
        if use_procrustes:
            ccs.append(dt.procrustes_wrapper(data1, data2))
        else:
            ccs.append(dt.canoncorr(data1, data2))
    ccs = np.array(ccs).T 

    return ccs


def get_ccs_upper_bound(side1df, epoch, area, n_components, calc_percentile = True, use_procrustes = False):
    n_iter = params.n_iter * 10

    #get data
    AllData1 = dt.get_data_array(side1df, epoch, area=area, model=n_components)
    n_shared_trial1 = AllData1.shape[2]
    trialList1 = np.arange(n_shared_trial1)

    #get ccs
    CCsU=[]
    for sessionData in AllData1:
        r = []
        for n in range(n_iter):
            params.rng.shuffle(trialList1)
            # non-overlapping randomised trials
            trial1 = trialList1[:n_shared_trial1//2]
            trial2 = trialList1[-(n_shared_trial1//2):]
            data1 = np.reshape(sessionData[:,trial1,:,:], (-1,n_components))
            data2 = np.reshape(sessionData[:,trial2,:,:], (-1,n_components))
            if use_procrustes:
                r.append(dt.procrustes_wrapper(data1, data2))
            else:
                r.append(dt.canoncorr(data1, data2))
        CCsU.append(r)
    CCsU = np.array(CCsU)
    if calc_percentile:
        CCsU = np.percentile(CCsU, 99, axis=1).T

    return CCsU

def get_ccs_lower_bound(side1df, side2df, area, n_components, len_trial, calc_percentile = True, use_procrustes = False):
    n_iter = params.n_iter * 10

    #get data
    AllData1 = _get_data_array(side1df, epoch_L=len_trial, area=area, model=n_components)
    AllData1_ =_get_data_array(side2df, epoch_L=len_trial, area=area, model=n_components)
    _,_, min_trials, min_time,_ = np.min((AllData1.shape,AllData1_.shape),axis=0)

    #get ccs
    CCsL=[]
    for sessionData1,sessionData2 in zip(AllData1,AllData1_):
        r = []
        for n in range(n_iter):
            sessionData1_sh = params.rng.permutation(sessionData1,axis=0)
            sessionData2_sh = params.rng.permutation(sessionData2,axis=0)

            data1 = np.reshape(sessionData1_sh[:,:min_trials,:min_time,:], (-1,n_components))
            data2 = np.reshape(sessionData2_sh[:,:min_trials,:min_time,:], (-1,n_components))
            if use_procrustes:
                r.append(dt.procrustes_wrapper(data1, data2))
            else:
                r.append(dt.canoncorr(data1, data2))
        CCsL.append(r)
    CCsL = np.array(CCsL)
    if calc_percentile:
        CCsL = np.percentile(CCsL, 1, axis=1).T
    return CCsL


@utility.report
def plot_cca(ax, ax_hist, allDFs, epoch, area, n_components, dataset = 'monkey', prep=False, use_procrustes = False):

    #get data
    if dataset == 'monkey':
        animals = 'monkeys'
        defs = monkey_defs
        pairFileList = get_paired_files_monkey(allDFs)
        n_animals = len(np.unique([df.monkey[0] for df in allDFs]))
    elif dataset == 'mouse':
        animals = 'mice'
        defs = mouse_defs
        pairFileList = get_paired_files_mouse(allDFs)
        n_animals = len(np.unique([df.mouse[0] for df in allDFs]))
    else:
        raise ValueError('dataset must be monkey or mouse')

    pair_side1df = [allDFs[i] for i,_ in pairFileList]
    pair_side2df = [allDFs[j] for _,j in pairFileList]
    side1df = allDFs

    #get ccs
    if prep:
        len_trial = int(np.round(np.diff(defs.WINDOW_prep)/defs.BIN_SIZE))
    else:
        len_trial = int(np.round(np.diff(defs.WINDOW_exec)/defs.BIN_SIZE))
    allCCs = get_ccs(pair_side1df, pair_side2df, epoch, area, n_components, use_procrustes = use_procrustes)
    CCsL = get_ccs_lower_bound(pair_side1df, pair_side2df, area, n_components, len_trial, use_procrustes = use_procrustes)
    CCsU = get_ccs_upper_bound(side1df, epoch, area, n_components, use_procrustes=use_procrustes)

    # plotting
    x_ = np.arange(1,n_components+1)
    utility.shaded_errorbar(ax, x_, allCCs, color=params.colors.MainCC, marker = 'o')
    utility.shaded_errorbar(ax, x_, CCsU, color=params.colors.UpperCC, marker = '<', ls='--')
    utility.shaded_errorbar(ax, x_, CCsL, color=params.colors.LowerCC, marker = '>', ls=':')

    ax.set_ylim([-.05,1])
    ax.set_xlim([.6,n_components+.6])
    ax.set_xlabel('Neural mode')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if use_procrustes:
        ax.set_ylabel('Procrustes correlation')
    else:
        ax.set_ylabel('Canonical correlation')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds([1,n_components])
    ax.spines['left'].set_bounds([0,1])
    ax.text(x=n_components, y=1, s= f'$n={CCsL.shape[1]}$ pairs of sessions\nacross ${n_animals}$ ${animals}$', ha='right', va='top', fontsize=mpl.rcParams['xtick.labelsize'])
    
    #plot the hist
    bins = np.arange(0,1,0.05)
    ax_hist.xaxis.set_visible(False)
    ax_hist.set_facecolor('None')
    ax_hist.spines['bottom'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['left'].set_bounds([0,1])
    ax_hist.set_ylim([-.05,1])
    ax_hist.hist(allCCs[:4,:].mean(axis=0), bins=bins, density=True, label=f'Across ($n={allCCs.shape[1]}$)', color=params.colors.MainCC, alpha=.8, orientation='horizontal')
    ax_hist.hist(CCsU[:4,:].mean(axis=0), bins=bins, density=True,label=f'Within ($n={CCsU.shape[1]}$)', color=params.colors.UpperCC, alpha=.8, orientation='horizontal')
    ax_hist.hist(CCsL[:4,:].mean(axis=0), bins=bins, density=True, label=f'Control ($n={CCsL.shape[1]}$)', color=params.colors.LowerCC, alpha=.8, orientation='horizontal')

    ax_hist.tick_params('y', direction='out')
    ax_hist.set_yticklabels([])
    ax_hist.legend(loc=(0,-.05))

    #stats ###################################
    allCCs_median = np.median(allCCs[:4,:].mean(axis=0))
    CCsU_median = np.median(CCsU[:4,:].mean(axis=0))
    CCsL_median = np.median(CCsL[:4,:].mean(axis=0))
    allCCs_mean = allCCs[:4,:].mean(axis=0)
    CCsU_mean = CCsU[:4,:].mean(axis=0)
    CCsL_mean = CCsL[:4,:].mean(axis=0)

    #calc stats
    ##for paired stats
    side1CCsU_mean = [CCsU_mean[i] for i,_ in pairFileList]
    side2CCsU_mean = [CCsU_mean[j] for _,j in pairFileList]
    allCCsU_mean = side1CCsU_mean + side2CCsU_mean
        
    compare_upper_stats = wilcoxon(np.tile(allCCs_mean,2), allCCsU_mean)
    compare_lower_stats = wilcoxon(allCCs_mean, CCsL_mean)

    ##for unpaired stats
    # compare_upper_stats = mannwhitneyu(allCCs_mean, CCsU_mean)
    # compare_lower_stats = mannwhitneyu(allCCs_mean, CCsL_mean)

    print("Across vs within:", compare_upper_stats)
    print("Across vs control:", compare_lower_stats)

    if params.annotate_stats:
        #annotate stats
        xmin, xmax = ax_hist.get_xlim()
        markerx = xmax+(xmax-xmin)*0.05
        linex = xmax+(xmax-xmin)*0.15
        textx = xmax+(xmax-xmin)*0.25
        line_kwargs = dict(linewidth = 0.5, color = 'k')
        text_kwargs = dict(ha='left', va='center')

        ax_hist.scatter(markerx, allCCs_median, color = params.colors.MainCC, marker = '<')
        ax_hist.scatter(markerx, CCsU_median, color = params.colors.UpperCC, marker = '<')
        ax_hist.scatter(markerx, CCsL_median, color = params.colors.LowerCC, marker = '<')

        ax_hist.plot([linex, linex], [allCCs_median, CCsU_median], **line_kwargs)
        ax_hist.plot([linex, linex], [allCCs_median, CCsL_median], linestyle = '--', **line_kwargs)
        
        ax_hist.text(textx, (allCCs_median + CCsU_median)/2, dt.get_signif_annot(compare_upper_stats[1]), **text_kwargs)
        ax_hist.text(textx, (allCCs_median + CCsL_median)/2, dt.get_signif_annot(compare_lower_stats[1]), **text_kwargs)

def plot_cca_for_ex(ax, example_dfs, epoch, area, n_components, dataset = 'monkey', prep = False):

    if dataset == 'monkey':
        animals = 'monkeys'
        defs = monkey_defs
    elif dataset == 'mouse':
        animals = 'mice'
        defs = mouse_defs
    else:
        raise ValueError('dataset must be monkey or mouse')
    
    #get data
    df1, df2 = example_dfs

    #get ccs
    if prep:
        len_trial = int(np.round(np.diff(defs.WINDOW_exec)/defs.BIN_SIZE))
    else:
        len_trial = int(np.round(np.diff(defs.WINDOW_prep)/defs.BIN_SIZE))
    allCCs = get_ccs(df1,df2, epoch, area, n_components)
    CCsL = get_ccs_lower_bound(df1,df2, area, n_components, len_trial)
    CCsU = get_ccs_upper_bound([df1,df2], epoch, area, n_components)

    # plotting
    x_ = np.arange(1,n_components+1)
    ax.plot(x_, allCCs, color=params.colors.MainCC, marker = 'o', label=f'Across ${animals}$')
    ax.plot(x_, CCsU[:,0], color=params.colors.UpperCC, marker = '<', ls='--', label=f'Within 1')
    ax.plot(x_, CCsU[:,1], marker = '<', ls='--', label=f'Within 2')
    ax.plot(x_, CCsL, color=params.colors.LowerCC, marker = '>', ls=':', label=f'Control')

    ax.set_ylim([-.05,1])
    ax.set_xlim([.6, n_components+.6])
    ax.set_xlabel('Neural mode')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.set_title(f'{defs.areas[2]} Alignment')
    ax.legend(loc=(.55,.67))
    ax.set_ylabel('Canonical correlation')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds([1, n_components])
    ax.spines['left'].set_bounds([0,1])

####################################################
# Rasters and positions
####################################################
@utility.report
def plot_fr_raster(df, gs, fig, targets, trial=12, area = 'M1'):
    axes = []
    data = []
    min_units = np.inf
    n_targets = len(targets)
    #example trial data for each target
    for tar in targets:
        df_ = pyal.select_trials(df, df.target_id==tar)
        df_ = pyal.remove_low_firing_neurons(df_, f'{area}_rates', 1)
        fr = df_[f'{area}_rates'][trial]
        data.append(fr)
    else:
        data = np.array(data)
    vmin = np.amin(data, axis= (0,1))
    vmax = np.amax(data, axis= (0,1))

    for j,tarData in enumerate(data):
        ax = fig.add_subplot(gs[j])
        axes.append(ax)
        tarData -= vmin
        tarData /= (vmax - vmin)
        ax.imshow(tarData.T, aspect='auto')
        ax.tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_title(r'$ \rightarrow $', rotation=(360/n_targets)*(j-3), pad=0.5) #TODO: remove?
    axes[0].set_ylabel(f'Units ($n={tarData.shape[1]}$)')
    return axes

@utility.report
def plot_pos_ex(df, gs, fig, targets, trial = 12, get_mean = True):
    axes = []
    for j, tar in enumerate(targets):
        df_ = pyal.select_trials(df, df.target_id==tar)
        data = df_.pos[trial]
        while np.isnan(data := df_.pos[trial]).sum()>0:
            trial +=1
        data -= np.mean(data,axis=0)
        if get_mean:
            data -= np.mean(data, axis=0, keepdims=True)
        ax = fig.add_subplot(gs[j])
        axes.append(ax)
        ax.plot(data[:,0], color='b', label='$X$')
        ax.plot(data[:,1], color='r', label='$Y$')
        if data.shape[1] == 3:
            ax.plot(data[:,2], color='g', label='$Z$')
        ax.tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    return axes

####################################################
# Decoding
####################################################
@utility.report
def monkey_target_decoding(allDFs):
    defs = monkey_defs
    classifier_model = GaussianNB

    within_score = {}
    aligned_score = {}
    unaligned_score = {}
    for i, df1 in enumerate(tqdm(allDFs)):
        if 'J' in df1.monkey[0]:
            continue  # remove Jaco from this analysis

        # within animal decoding ######
        AllData = dt.get_data_array([df1], defs.prep_epoch, area=defs.areas[2], model=defs.n_components)
        # adding history
        AllData = dt.add_history_to_data_array(AllData,defs.MAX_HISTORY)
        AllData1 = AllData[0,...]
        _,n_trial,n_time,n_comp = AllData1.shape
        # resizing
        X1 = AllData1.reshape((-1,n_comp*n_time))
        AllTar = np.repeat(np.arange(defs.n_targets),n_trial)
        # train the decoder
        _score=cross_val_score(classifier_model(),X1,AllTar,scoring='accuracy', cv=10).mean()
        within_score[df1.session[0]] = _score

        # across animal decoding ######
        aligned_score[df1.session[0]] = {}
        unaligned_score[df1.session[0]] = {}
        for j, df2 in enumerate(tqdm(allDFs)):
            if j <= i: continue
            if df1.monkey[0] == df2.monkey[0]: continue
            if 'Chewie' in df1.monkey[0] and 'Chewie' in df2.monkey[0]: continue
            if 'J' in df1.monkey[0] or 'J' in df2.monkey[0]: continue  # remove Jaco from this analysis

            AllData = dt.get_data_array([df1,df2], defs.prep_epoch, area=defs.areas[2], model=defs.n_components)

            # adding history
            AllData = dt.add_history_to_data_array(AllData, defs.MAX_HISTORY)
            AllData1 = AllData[0,...] 
            AllData2 = AllData[1,...]
            _,n_trial,n_time,n_comp = AllData1.shape

            # resizing
            X1 = AllData1.reshape((-1,n_comp))
            X2 = AllData2.reshape((-1,n_comp))

            # aligned ###
            *_,U,V = dt.canoncorr(X1, X2, fullReturn=True)
            U = U.reshape((-1,n_comp*n_time))
            V = V.reshape((-1,n_comp*n_time))
            AllTar = np.repeat(np.arange(defs.n_targets),n_trial)
            trial_index = np.arange(len(AllTar))
            params.rng.shuffle(trial_index)
            X_train, Y_train = U[trial_index,:], AllTar[trial_index]
            params.rng.shuffle(trial_index)
            X_test, Y_test   = V[trial_index,:], AllTar[trial_index]

            # train the decoder
            classifier = classifier_model()
            classifier.fit(X_train, Y_train)
            # test the decoder
            _score = classifier.score(X_test,Y_test)
            aligned_score[df1.session[0]][df2.session[0]]=_score

            # unaligned ###
            X1 = X1.reshape((-1,n_comp*n_time))
            X2 = X2.reshape((-1,n_comp*n_time))
            AllTar = np.repeat(np.arange(defs.n_targets),n_trial)
            trial_index = np.arange(len(AllTar))
            params.rng.shuffle(trial_index)
            X_train, Y_train = X1[trial_index,:], AllTar[trial_index]
            params.rng.shuffle(trial_index)
            X_test, Y_test   = X2[trial_index,:], AllTar[trial_index]
            # train the decoder
            classifier = classifier_model()
            classifier.fit(X_train, Y_train)
            # test the decoder
            _score = classifier.score(X_test,Y_test)
            unaligned_score[df1.session[0]][df2.session[0]]=_score

    return within_score, aligned_score, unaligned_score

def within_animal_decoding(df1, defs, epoch, area, n_components, custom_r2_func = None, normalize_pos = False):
    # get data
    AllData, AllVel = get_data_array_and_pos([df1], epoch = epoch, area = area, n_components = n_components, normalize_pos = normalize_pos)
    # adding history
    AllData = dt.add_history_to_data_array(AllData,defs.MAX_HISTORY)
    AllData = AllData[...,defs.MAX_HISTORY:,:]
    AllVel = AllVel[...,defs.MAX_HISTORY:,:]
    AllData1 = AllData[0,...]
    AllVel1 = AllVel[0,...]
    *_,n_time,n_comp = AllData1.shape
    # resizing
    X1 = AllData1.reshape((-1,n_time,n_comp))
    AllVel1 = AllVel1.reshape((-1,n_time,defs.output_dims))
    
    # train and test decoders
    fold_score =[]
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X1[:,0,0]):
        x_train, x_test = X1[train_index,...], X1[test_index,...]
        y_train, y_test = AllVel1[train_index,...], AllVel1[test_index,...]

        lstm_model = lstm.LSTMDecoder(input_dims=X1.shape[-1], output_dims=defs.output_dims)
        lstm_model.fit(x_train=x_train, y_train=y_train)
        pred, label = lstm_model.predict(x_test, y_test)
        if custom_r2_func is not None:
            fold_score.append(custom_r2_func(pred, label))
        else:
            fold_score.append(lstm_model.score)
    fold_score = np.median(fold_score)

    return fold_score

def across_animal_decoding(df1, df2, defs, epoch, area, n_components, custom_r2_func = None, normalize_pos = False):
    # get data
    AllData, AllVel = get_data_array_and_pos([df1,df2], epoch = epoch, area = area, n_components = n_components, normalize_pos=normalize_pos)
    # adding history
    AllData = dt.add_history_to_data_array(AllData,defs.MAX_HISTORY)
    AllData = AllData[...,defs.MAX_HISTORY:,:]
    AllVel = AllVel[...,defs.MAX_HISTORY:,:]

    AllData1 = AllData[0,...]
    AllData2 = AllData[1,...]
    AllVel1 = AllVel[0,...]
    AllVel2 = AllVel[1,...]
    # resizing
    *_,n_time,n_comp = AllData1.shape

    X1 = AllData1.reshape((-1,n_comp))
    X2 = AllData2.reshape((-1,n_comp))
    AllVel1 = AllVel1.reshape((-1,n_time,defs.output_dims))
    AllVel2 = AllVel2.reshape((-1,n_time,defs.output_dims))

    # train the aligned ###
    *_,U,V = dt.canoncorr(X1, X2, fullReturn=True)
    U = U.reshape((-1,n_time,n_comp))
    V = V.reshape((-1,n_time,n_comp))
    X1 = X1.reshape((-1,n_time,n_comp))
    X2 = X2.reshape((-1,n_time,n_comp))

    lstm_model = lstm.LSTMDecoder(input_dims=U.shape[-1], output_dims=defs.output_dims)
    lstm_model.fit(x_train=U, y_train=AllVel1)
    pred, label = lstm_model.predict(V, AllVel2)
    if custom_r2_func is not None:
        aligned_score = custom_r2_func(pred, label).mean()
    else:
        aligned_score = lstm_model.score.mean()

    # train the unaligned ###
    lstm_model1 = lstm.LSTMDecoder(input_dims=X1.shape[-1], output_dims=defs.output_dims)
    lstm_model1.fit(x_train=X1, y_train=AllVel1)
    pred1, label1 = lstm_model1.predict(X2, AllVel2)
    lstm_model2 = lstm.LSTMDecoder(input_dims=X1.shape[-1], output_dims=defs.output_dims)
    lstm_model2.fit(x_train=X2, y_train=AllVel2)
    pred2, label2 = lstm_model2.predict(X1, AllVel1)

    if custom_r2_func is not None:
        unaligned_score = (custom_r2_func(pred1, label1).mean() + custom_r2_func(pred2, label2).mean()) / 2
    else:
        unaligned_score = (lstm_model1.score.mean() + lstm_model2.score.mean()) / 2

    return aligned_score, unaligned_score

@utility.report
def mouse_decoding(allDFs, epoch, area, custom_r2_func = None, normalize_pos = False):
    defs = mouse_defs
    
    within_score = {}
    aligned_score = {}
    unaligned_score = {}
    for i, df1 in enumerate(tqdm(allDFs)):
        animal1 = df1.mouse[0]

        within_score_ = within_animal_decoding(df1, defs, epoch, area, defs.n_components, custom_r2_func, normalize_pos)
        within_score[df1.file[0]] = within_score_

        aligned_score[df1.file[0]] = {}
        unaligned_score[df1.file[0]] = {}
        for j, df2 in enumerate(allDFs):
            if j < i: continue
            animal2 = df2.mouse[0]
            if animal1 == animal2: continue
            
            aligned_score_, unaligned_score_ = across_animal_decoding(df1, df2, defs, epoch, area, defs.n_components, custom_r2_func, normalize_pos)
         
            aligned_score[df1.file[0]][df2.file[0]] = aligned_score_
            unaligned_score[df1.file[0]][df2.file[0]] = unaligned_score_
    return within_score, aligned_score, unaligned_score

def monkey_decoding(allDFs, epoch, area, redo = False, n_components = monkey_defs.n_components, custom_r2_func = None, normalize_pos = False):
    defs = monkey_defs

    within_score = {}
    aligned_score = {}
    unaligned_score = {}
    for i, df1 in enumerate(tqdm(allDFs)):
        pathPickle = params.root / 'monkey-pickles' / f'{df1.session[0]}_{n_components}_within.p'

        # within animal decoding ######
        if os.path.exists(pathPickle) and not redo:
            # get saved data
            with open(pathPickle,"rb") as f:
                within_score[df1.session[0]] = pickle.load(f)
        else:
            within_score_ = within_animal_decoding(df1, defs, epoch, area, n_components, custom_r2_func, normalize_pos)
            within_score[df1.session[0]] = within_score_
            #save data
            with open(pathPickle, 'wb') as f:
                pickle.dump(within_score[df1.session[0]], f)
                f.close()

        # across animal decoding ######
        aligned_score[df1.session[0]] = {}
        unaligned_score[df1.session[0]] = {}
        for j, df2 in enumerate(tqdm(allDFs)):
            if j <= i: continue
            if df1.monkey[0] == df2.monkey[0]: continue
            if 'Chewie' in df1.monkey[0] and 'Chewie' in df2.monkey[0]: continue
            alignedPickle = params.root / 'monkey-pickles' /  f'{df1.session[0]}-{df2.session[0]}_{n_components}_aligned.p'
            unalignedPickle = params.root / 'monkey-pickles' /  f'{df1.session[0]}-{df2.session[0]}_{n_components}_unaligned.p'
            if os.path.exists(alignedPickle) and os.path.exists(unalignedPickle) and not redo:
                #get saved data
                with open(alignedPickle,"rb") as f:
                    aligned_score[df1.session[0]][df2.session[0]] = pickle.load(f)
                    f.close()
                with open(unalignedPickle,"rb") as f:
                    unaligned_score[df1.session[0]][df2.session[0]] = pickle.load(f)
                    f.close()
            else:
                aligned_score_, unaligned_score_ = across_animal_decoding(df1, df2, defs, epoch, area, n_components, custom_r2_func, normalize_pos)
             
                aligned_score[df1.session[0]][df2.session[0]]=aligned_score_  
                unaligned_score[df1.session[0]][df2.session[0]]=unaligned_score_

                #save data
                with open(alignedPickle, 'wb') as f:
                    pickle.dump(aligned_score_, f)
                    f.close()
                with open(unalignedPickle, 'wb') as f:
                    pickle.dump(unaligned_score_, f)
                    f.close()

    return within_score, aligned_score, unaligned_score

def plot_decoding(ax, allDFs, epoch, area, target = False, redo = False, dataset = 'monkey', color_by_behav_corr = False, custom_r2_func = None, normalize_pos = False):

    if target:
        within_score, aligned_score, unaligned_score = monkey_target_decoding(allDFs, epoch, area)
    else:
        if dataset == 'monkey':
            within_score, aligned_score, unaligned_score = monkey_decoding(allDFs, epoch, area, redo, custom_r2_func, normalize_pos)
        elif dataset == 'mouse':
            within_score, aligned_score, unaligned_score = mouse_decoding(allDFs, epoch, area, custom_r2_func, normalize_pos)

    pop_within = np.array(list(within_score.values()))
    pop_aligned = np.array([val for key in aligned_score for val in aligned_score[key].values()])
    pop_unaligned = np.array([val for key in unaligned_score for val in unaligned_score[key].values()])

    ax.errorbar(1, pop_aligned.mean(), np.std(pop_aligned), label='Across\n' r'(\textit{aligned})',
                color=params.colors.MainCC, fmt='-o', capsize=1.5)    
    ax.errorbar(0, pop_unaligned.mean(), np.std(pop_unaligned), label='Across\n' r'(\textit{unaligned})',
                color=params.colors.LowerCC, fmt='-o', capsize=1.5)
    ax.errorbar(2, pop_within.mean(), np.std(pop_within), label='Within',
                color=params.colors.UpperCC, fmt='-o', capsize=1.5)

    if color_by_behav_corr:
        colormap = 'cool'
        if dataset == 'mouse':
            across_corrs = trim_across_mouse_corr(allDFs)
        else:
            across_corrs = trim_across_monkey_corr(allDFs)

        pop_behav_corr = []
        for file1, nested_dict in aligned_score.items():
            wi_val1 = within_score[file1]
            for file2, al_val in nested_dict.items():
                behav = np.array(across_corrs[file1][file2])
                behav = np.mean(behav[behav>params.Behav_corr_TH])
                pop_behav_corr.append(behav)

        min = np.min(pop_behav_corr)
        max = np.max(pop_behav_corr)
        norm = mpl.colors.Normalize(vmin=min, vmax=max+(max-min)*0.1)

    unal_vals = []
    al_vals = []
    wi_vals = []
    for file1, nested_dict in aligned_score.items():
        wi_val1 = within_score[file1]
        for file2, al_val in nested_dict.items():
            wi_val2 = within_score[file2]
            unal_val = unaligned_score[file1][file2]

            if color_by_behav_corr:
                behav = np.array(across_corrs[file1][file2])
                behav = np.mean(behav[behav>params.Behav_corr_TH])
                ax.plot([0,1,2], [unal_val, al_val, wi_val1],
                        color = mpl.colormaps[colormap](norm(behav)), lw=.2, zorder=6, marker = 'o', ms=.1, alpha=.7)
                ax.plot([1,2], [al_val, wi_val2],
                        color = mpl.colormaps[colormap](norm(behav)), lw=.2, zorder=6, marker = 'o', ms=.1, alpha=.7)
            else: 
                ax.plot([0,1,2], [unal_val, al_val, wi_val1],
                        color='gray', lw=.2, zorder=6, marker = 'o', ms=.1, alpha=.2)
                ax.plot([1,2], [al_val, wi_val2],
                        color='gray', lw=.2, zorder=6, marker = 'o', ms=.1, alpha=.2)
                
            #for stats
            unal_vals.append(unal_val)
            al_vals.append(al_val)
            wi_vals.append(wi_val1)
            wi_vals.append(wi_val2)
    
    if color_by_behav_corr:
        plt.colorbar(cm.ScalarMappable(norm = norm, cmap = colormap), ax = ax)

    ax.set_xlim([-0.2,2.2])
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['Across\n' r'(\textit{unaligned})',
                        'Across\n' r'(\textit{aligned})',
                        'Within'])
    if target:
        ax.set_ylabel('Classification accuracy')
    else:
        ax.set_ylabel('Prediction accuracy ($R^2$)')
    ax.set_ylim([-.05,1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds([0,2])
    ax.spines['left'].set_bounds([0,1])

    #stats ########################################
    #calc stats
    ##for paired stats
    compare_upper_stats = wilcoxon(np.repeat(al_vals,2), wi_vals)
    compare_lower_stats = wilcoxon(al_vals, unal_vals)

    print("Across vs within:", compare_upper_stats)
    print("Across vs control:", compare_lower_stats)

    #annotate stats
    if params.annotate_stats:
        ymin, ymax = ax.get_ylim()
        liney = ymax*0.95
        texty = ymax*1
        line_kwargs = dict(linewidth = 0.5, color = 'k')
        text_kwargs = dict(ha='left', va='center')

        ax.plot([0,1], [liney, liney], **line_kwargs)
        ax.plot([1,2], [liney, liney], linestyle = '--', **line_kwargs)

        ax.text(0.5, texty, dt.get_signif_annot(compare_upper_stats[1]), **text_kwargs)
        ax.text(1.5, texty, dt.get_signif_annot(compare_lower_stats[1]), **text_kwargs)


####################################################
# Behavioral correlation vs CCA
####################################################
def del_nan(a,b):
    a_bad = np.isnan(a)
    b_bad = np.isnan(b)
    bad = np.logical_or(a_bad,b_bad)
    good = np.logical_not(bad)
    
    return a[good].reshape(-1,1), b[good]

def trim_within_mouse_corr(allDF:list[pd.DataFrame], score = r_regression):
    trim_within_corrs = {}
    for df__ in allDF:
        df = pyal.restrict_to_interval(df__, epoch_fun=mouse_defs.exec_epoch)
        trim_within_corrs[df.file[0]] = []
        targets = np.unique(df.target_id)
        for target in targets:
            df_ = pyal.select_trials(df, df.target_id == target)
            for i, pos1 in enumerate(df_.hTrjB):
                for j, pos2 in enumerate(df_.hTrjB):
                    if j<=i: continue
                    r = [float(score(*del_nan(aa,bb))) for aa,bb in zip(pos1.T,pos2.T)]
                    trim_within_corrs[df_.file[0]].append(np.mean(np.abs(r)))

    return trim_within_corrs


def trim_across_mouse_corr(allDF:list[pd.DataFrame], score = r_regression):
    trim_across_corrs = {}
    across_good = across_bad = 0
    for dfi, df1__ in enumerate(allDF):
        df1 = pyal.restrict_to_interval(df1__, epoch_fun=mouse_defs.exec_epoch)
        targets = np.unique(df1.target_id)
        trim_across_corrs[df1.file[0]]={}
        for dfj, df2__ in enumerate(allDF):
            df2 = pyal.restrict_to_interval(df2__, epoch_fun=mouse_defs.exec_epoch)
            trim_across_corrs[df2.file[0]] = {} if df2.file[0] not in trim_across_corrs.keys() else trim_across_corrs[df2.file[0]]
            if dfj <= dfi: continue
            trim_across_corrs[df1.file[0]][df2.file[0]]=[]
            for target in targets:
                df1_ = pyal.select_trials(df1, df1.target_id == target)
                df2_ = pyal.select_trials(df2, df2.target_id == target)
                for i, pos1 in enumerate(df1_.hTrjB):
                    for j, pos2 in enumerate(df2_.hTrjB):
                        r = [float(score(*del_nan(aa,bb))) for aa,bb in zip(pos1.T,pos2.T)]
                        trim_across_corrs[df1_.file[0]][df2_.file[0]].append(np.mean(np.abs(r)))
                        across_good += 1

        # make the across correlations symmetrical!
        for  df2_file, val in trim_across_corrs[df1_.file[0]].items():
            trim_across_corrs[df2_file][df1_.file[0]] = val

    return trim_across_corrs

def trim_within_monkey_corr(allDF:list[pd.DataFrame], score = r_regression, redo = False):
    within_corrs = {}
    for df__ in allDF:
        within_corrs[df__.session[0]] = []
        pathPickle = params.root / 'monkey-pickles' / f'{df__.session[0]}_within_{score.__name__}.p'
        if os.path.exists(pathPickle) and not redo:
            with open(pathPickle,"rb") as f:
                result=pickle.load(f)
            within_corrs[df__.session[0]] = result
            continue
        else:
            df = pyal.restrict_to_interval(df__, epoch_fun=monkey_defs.exec_epoch)
            targets = np.unique(df.target_id)
            for target in targets:
                df_ = pyal.select_trials(df, df.target_id == target)
                for i, pos1 in enumerate(df_.pos):
                    a = pos1
                    for j, pos2 in enumerate(df_.pos):
                        if j<=i: continue
                        b = pos2
                        r = [float(score(aa.reshape(-1,1),bb)) for aa,bb in zip(a.T,b.T)]
                        within_corrs[df_.session[0]].append(np.mean(np.abs(r)))
        with open(pathPickle, 'wb') as f:
            pickle.dump(within_corrs[df_.session[0]], f)
            f.close()
    
    return within_corrs

def trim_across_monkey_corr(allDF:list[pd.DataFrame], score = r_regression, redo=False):
    across_corrs = {}
    #for each session
    for dfi, df1__ in enumerate(allDF):
        df1 = pyal.restrict_to_interval(df1__, epoch_fun=monkey_defs.exec_epoch)
        targets = np.unique(df1.target_id)
        across_corrs[df1.session[0]]={}

        #compare to each session
        for dfj, df2__ in enumerate(allDF):
            #save results in dict
            pathPickle = params.root / 'monkey-pickles' / f'{df1__.session[0]}_{df2__.session[0]}_across_{score.__name__}.p'
            if os.path.exists(pathPickle) and not redo:
                with open(pathPickle,"rb") as f:
                    result=pickle.load(f)
                across_corrs[df1__.session[0]][df2__.session[0]] = result
                across_corrs[df2__.session[0]] = {} if df2__.session[0] not in across_corrs.keys() else across_corrs[df2__.session[0]]
                continue
            df2 = pyal.restrict_to_interval(df2__, epoch_fun=monkey_defs.exec_epoch)
            across_corrs[df2.session[0]] = {} if df2.session[0] not in across_corrs.keys() else across_corrs[df2.session[0]]
            if dfj <= dfi: continue
            across_corrs[df1.session[0]][df2.session[0]]=[]

            #for each target
            for target in targets:
                df1_ = pyal.select_trials(df1, df1.target_id == target)
                df2_ = pyal.select_trials(df2, df2.target_id == target)
                #correlate pairs of reaches
                for i, pos1 in enumerate(df1_.pos):
                    for j, pos2 in enumerate(df2_.pos):
                        r = [float(score(aa.reshape(-1,1),bb)) for aa,bb in zip(pos1.T,pos2.T)]
                        across_corrs[df1_.session[0]][df2_.session[0]].append(np.mean(np.abs(r)))
            
            with open(pathPickle, 'wb') as f:
                pickle.dump(across_corrs[df1_.session[0]][df2_.session[0]], f)
                f.close()

        # make the across correlations symmetrical!
        for  df2_session, val in across_corrs[df1__.session[0]].items():
            across_corrs[df2_session][df1__.session[0]] = val

    return across_corrs

@utility.report
def plot_cca_corr(ax, allDFs, epoch, area, n_components, dataset='monkey'):

    #get behavioral correlation for paired reaches
    if dataset == 'monkey':
        across_corrs = trim_across_monkey_corr(allDFs)
        pairFileList = get_paired_files_monkey(allDFs)
        color = params.colors.MonkeyPts
        label = 'Monkeys'
    elif dataset == 'mouse':
        across_corrs = trim_across_mouse_corr(allDFs)
        pairFileList = get_paired_files_mouse(allDFs)
        color = params.colors.MousePts
        label = 'Mice'
    else:
        raise ValueError('dataset must be monkey or mouse')
    
    #get data for neural modes
    side1df = [allDFs[i] for i,_ in pairFileList]
    side2df = [allDFs[j] for _,j in pairFileList]

    # get ccs
    allCCs = get_ccs(side1df, side2df, epoch, area, n_components) # n_pairs x n_components

    CC_corr=[]
    # for each pair of sessions, save data
    for i, (k,l) in enumerate(pairFileList):
        behav = np.array(across_corrs[allDFs[k].session[0]][allDFs[l].session[0]])
        behav = behav[behav>params.Behav_corr_TH]
        CC_corr.append((allCCs[:4, i].mean() , np.mean(behav)))
    CC_corr = np.array(CC_corr)
    
    #plotting
    ax.scatter(CC_corr[:,1],CC_corr[:,0], color=color, label=label, zorder=0)
    ax.set_xlabel('Behavioural correlation')
    ax.set_ylabel('Canonical correlation')
    ax.set_ylim([.53,.85])
    ax.spines['left'].set_bounds([.55,.85])
    ax.set_xlim([.69,.95])
    ax.spines['bottom'].set_bounds([.7,.95])
    ax.legend(loc=(0,.8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('$%0.2f$'))

    return CC_corr[:,1], CC_corr[:,0]
    

####################################################
# Other
####################################################
def _get_data_array(data_list: list[pd.DataFrame], epoch_L: int =None , area: str ='M1', model=None) -> np.ndarray:
    "Similat to `get_data_array` only returns an apoch of length `epoch_L` randomly chosen along each trial"
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
            all_id = df__.trial_id.to_numpy()
            # to guarantee shuffled ids
            rng.shuffle(all_id)
            # select the right number of trials to each target
            df__ = pyal.select_trials(df__, lambda trial: trial.trial_id in all_id[:n_shared_trial])
            for trial, trial_rates in enumerate(df__._pca):
                time_idx = rng.integers(trial_rates.shape[0]-epoch_L)
                trial_data = trial_rates[time_idx:time_idx+epoch_L,:]
                AllData[session,targetIdx,trial, :, :] = trial_data

    return AllData

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
    target_ids = np.unique(data_list[0].target_id)
    n_targets = len(target_ids)
    for df in data_list:
        for target in range(n_targets):
            df_ = pyal.select_trials(df, df.target_id == target)
            n_shared_trial = np.min((df_.shape[0], n_shared_trial))

    n_shared_trial = int(n_shared_trial)

    # finding the number of timepoints
    df_ = pyal.restrict_to_interval(df_,epoch_fun=epoch)
    n_timepoints = int(df_[field][0].shape[0])
    n_outputs = df_.pos[0].shape[1]

    # pre-allocating the data matrix
    AllData = np.empty((len(data_list), n_targets, n_shared_trial, n_timepoints, n_components))
    AllVel  = np.empty((len(data_list), n_targets, n_shared_trial, n_timepoints, n_outputs))
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