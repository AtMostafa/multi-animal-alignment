
import params
import numpy as np
import pyaldata as pyal
from tools import utilityTools as utility

@utility.report
def plot_fr_raster(df, gs, fig, targets, trial=12, area = 'M1'):
    axes = []
    data = []
    n_targets = len(targets)
    #example trial data for each target
    for tar in targets:
        df_ = pyal.select_trials(df, df.target_id==tar)
        # df_ = pyal.remove_low_firing_neurons(df_, f'{area}_rates', 1)
        fr = df_[f'{area}_rates'][trial]
        data.append(fr)
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