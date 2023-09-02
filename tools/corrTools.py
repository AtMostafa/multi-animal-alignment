import params
import os
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import pyaldata as pyal
from scipy.stats import wilcoxon
import pandas as pd
import pickle
from sklearn.feature_selection import r_regression

from tools import dataTools as dt
from tools import utilityTools as utility
from tools import ccaTools as cca
monkey_defs = params.monkey_defs
mouse_defs = params.mouse_defs

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
        pairFileList = dt.get_paired_files_monkey(allDFs)
        color = params.colors.MonkeyPts
        label = 'Monkeys'
    elif dataset == 'mouse':
        across_corrs = trim_across_mouse_corr(allDFs)
        pairFileList = dt.get_paired_files_mouse(allDFs)
        color = params.colors.MousePts
        label = 'Mice'
    else:
        raise ValueError('dataset must be monkey or mouse')
    
    #get data for neural modes
    side1df = [allDFs[i] for i,_ in pairFileList]
    side2df = [allDFs[j] for _,j in pairFileList]

    # get ccs
    allCCs = cca.get_ccs(side1df, side2df, epoch, area, n_components) # n_pairs x n_components

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
    