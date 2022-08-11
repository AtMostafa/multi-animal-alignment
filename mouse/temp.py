import os, sys, pathlib
from pprint import pprint
import gc
import pickle
from importlib import reload
import logging, warnings
logging.basicConfig(level=logging.ERROR)
warnings.simplefilter("ignore")


from tqdm import tqdm
import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import PCA
import scipy.linalg as linalg
import scipy.stats as stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score, KFold

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection

import torch

import pyaldata as pyal

try:
    nbPath = pathlib.Path.cwd()
    RepoPath = nbPath.parent
    os.chdir(RepoPath)

    from tools import utilityTools as utility
    from tools import dataTools as dt
    from tools import lstm
    import params
    mouse_defs = params.mouse_defs
    defs = mouse_defs

    set_rc =  params.set_rc_params
    set_rc()
    root = params.root
    reload(dt)
    reload(defs)
    reload(params)
finally:
    os.chdir(nbPath)
    
    
def get_full_mouse_data():
    defs = mouse_defs
    
    animalList = ['mouse-data']
    animalFiles = []
    for animal in animalList:
        animalFiles.extend(utility.find_file(root / animal, 'mat'))

    AllDFs=[]
    for fname in animalFiles:
        df = dt.load_pyal_data(fname)
        df['mouse'] = fname.split(os.sep)[-1][fname.split(os.sep)[-1].find('WR'):].split('_')[0]
        df['file'] = fname.split(os.sep)[-1]
        df = defs.prep_general_mouse(df)
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
    
    
def plot_m1_decoding(AllDFs):
    defs = mouse_defs
    
    reg_scores = []
    for i, df in enumerate(AllDFs):
        AllData, AllVel = defs.get_data_array_and_vel([df], defs.exec_epoch_decode, area=defs.areas[0],
                                                      n_components=defs.n_components)
        # adding history
        AllData = dt.add_history_to_data_array(AllData, defs.MAX_HISTORY)
        AllData = AllData[...,defs.MAX_HISTORY:,:]
        AllVel = AllVel[...,defs.MAX_HISTORY:,:]

        *_,n_time,n_comp = AllData.shape
        AllData1 = AllData[0,...]
        AllVel1 = AllVel[0,...]

        # resizing
        X1 = AllData1.reshape((-1, n_time, n_comp))
        AllVel1 = AllVel1.reshape((-1,n_time,3))

        fold_score =[]
        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(X1[:,0,0]):
            x_train, x_test = X1[train_index,...], X1[test_index,...]
            y_train, y_test = AllVel1[train_index,...], AllVel1[test_index,...]

            lstm_model = lstm.LSTMDecoder(input_dims=X1.shape[-1], output_dims=3)
            lstm_model.fit(x_train=x_train, y_train=y_train)
            lstm_model.predict(x_test, y_test)
            fold_score.append(lstm_model.score)
        fold_score = np.array(fold_score)
        reg_scores.append(np.median(fold_score, axis=0).mean())

    pop_score_day = np.array(reg_scores)


    #=========================
    # pairIndex_across = []
    # for i, df1 in enumerate(AllDFs):
    #     animal1 = df1.mouse[0]
    #     pairIndex_across.append((i,[]))
    #     for j, df2 in enumerate(AllDFs):
    #         if j<i: continue
    #         animal2 = df2.mouse[0]
    #         if animal1 == animal2: continue
    #         pairIndex_across[-1][1].append(j)
    # pairIndex_across = [(i,j) for i,jList in pairIndex_across for j in jList]
    
    reg_scores_across = []
    # for _, (id1, testId) in enumerate(pairIndex_across):
    #     AllData, AllVel = defs.get_data_array_and_vel([AllDFs[id1],AllDFs[testId]], defs.exec_epoch_decode,
    #                                                   area=defs.areas[0], n_components=defs.n_components)

    #     # adding history
    #     AllData = dt.add_history_to_data_array(AllData, defs.MAX_HISTORY)
    #     AllData = AllData[...,defs.MAX_HISTORY:,:]
    #     AllVel = AllVel[...,defs.MAX_HISTORY:,:]


    #     AllData1 = AllData[0,...]
    #     AllData2 = AllData[1,...]
    #     AllVel1 = AllVel[0,...]
    #     AllVel2 = AllVel[1,...]
        
    #     # resizing
    #     _,n_trial,n_time,n_comp = np.min(np.array((AllData1.shape,AllData2.shape)),axis=0)
    #     X1 = AllData1.reshape((-1,n_comp))
    #     X2 = AllData2.reshape((-1,n_comp))
    #     AllVel1 = AllVel1.reshape((-1,n_time,3))
    #     AllVel2 = AllVel2.reshape((-1,n_time,3))

    #     *_,U,V = dt.canoncorr(X1, X2, fullReturn=True)
    #     U = U.reshape((-1,n_time,n_comp))
    #     V = V.reshape((-1,n_time,n_comp))

    #     lstm_model = lstm.LSTMDecoder(input_dims=U.shape[-1], output_dims=3)
    #     lstm_model.fit(x_train=U, y_train=AllVel1)
    #     lstm_model.predict(V, AllVel2)
    #     reg_scores_across.extend(lstm_model.score.tolist())

    pop_score_across = np.array(reg_scores_across)

    #================================
    reg_latent_scores = []
    # for id1, testId in pairIndex_across:
    #     AllData, AllVel = defs.get_data_array_and_vel([AllDFs[id1],AllDFs[testId]], defs.exec_epoch_decode,
    #                                                   area=defs.areas[0], n_components=defs.n_components)

    #     # adding history
    #     AllData = dt.add_history_to_data_array(AllData, defs.MAX_HISTORY)
    #     AllData = AllData[...,defs.MAX_HISTORY:,:]
    #     AllVel = AllVel[...,defs.MAX_HISTORY:,:]

    #     AllData1 = AllData[0,...]
    #     AllData2 = AllData[1,...]
    #     AllVel1 = AllVel[0,...]
    #     AllVel2 = AllVel[1,...]
    #     # resizing
    #     _,n_trial,n_time,n_comp = np.min(np.array((AllData1.shape, AllData2.shape)),axis=0)
    #     X1 = AllData1.reshape((-1,n_time,n_comp))
    #     X2 = AllData2.reshape((-1,n_time,n_comp))
    #     AllVel2 = AllVel2.reshape((-1,n_time,3))
    #     AllVel1 = AllVel1.reshape((-1,n_time,3))
    #     crs_val_factor = int(0.9 * X1.shape[0])

    #     # train the decoder
    #     U,V = X1, X2
    #     lstm_model = lstm.LSTMDecoder(input_dims=U.shape[-1], output_dims=3)
    #     lstm_model.fit(x_train=U[:crs_val_factor,...], y_train=AllVel1[:crs_val_factor,...])
    #     lstm_model.predict(V[:crs_val_factor,...], AllVel2[:crs_val_factor,...])
    #     reg_latent_scores.extend(lstm_model.score.tolist())
    pop_latent_score = np.array(reg_latent_scores)


    return pop_score_across, pop_latent_score, pop_score_day


plt.close('all')
set_rc()
fig=plt.figure(dpi=100)
ax = fig.add_subplot()


allDFs_M1, _ = get_full_mouse_data()

aligned, unaligned, within = plot_m1_decoding(allDFs_M1)


plt.plot(within, label=['x','y','z'])
plt.legend()
plt.savefig('dummy_name.png')


print()
