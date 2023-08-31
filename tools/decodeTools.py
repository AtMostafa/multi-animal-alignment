import params
import os
import numpy as np
import matplotlib as mpl
from scipy.stats import wilcoxon
import pickle
from tqdm import tqdm
from sklearn.model_selection import KFold, cross_val_score
from tools import lstm
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tools import dataTools as dt
from tools import corrTools as ct
from tools import utilityTools as utility
monkey_defs = params.monkey_defs
mouse_defs = params.mouse_defs

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
            *_,U,V = cca.canoncorr(X1, X2, fullReturn=True)
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
    AllData, AllVel = dt.get_data_array_and_pos([df1], epoch = epoch, area = area, n_components = n_components, normalize_pos = normalize_pos)
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
    AllData, AllVel = dt.get_data_array_and_pos([df1,df2], epoch = epoch, area = area, n_components = n_components, normalize_pos=normalize_pos)
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
    *_,U,V = cca.canoncorr(X1, X2, fullReturn=True)
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
            across_corrs = ct.trim_across_mouse_corr(allDFs)
        else:
            across_corrs = ct.trim_across_monkey_corr(allDFs)

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

