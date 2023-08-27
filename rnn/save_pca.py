#%%
'''
Save latent dynamics (PCs) from previously trained network. Used to constrain CCs during training.
'''
#%%
from random import sample
import pyaldata as pyal
from params import rnn_defs
from tools import simTools as st
from sklearn.decomposition import PCA
import numpy as np
import sys
from params import rnn_defs

# %%
seed = int(sys.argv[1])
sim_number = int(sys.argv[2])
rnn_epoch = rnn_defs.exec_epoch

# get and process testing data and output from previously trained network
df = st.get_processed_pyaldata(seed, sim_number)
move_onsets = df.idx_movement_on.values
df = pyal.restrict_to_interval(df,epoch_fun = rnn_epoch)

# apply PCA
rates = np.concatenate(df['MCx_rates'].values, axis=0)
rates_model = PCA(rnn_defs.n_components).fit(rates)
df = pyal.apply_dim_reduce_model(df, rates_model, 'MCx_rates', 'MCx_pca')
df = df.groupby('target_id').head(rnn_defs.BATCH_SIZE/rnn_defs.n_targets)
pca = np.concatenate(df['MCx_pca'].values, axis=0)

# save data
dict={
    'pca': pca,
    'move_onsets': move_onsets, #movement on based on threshold=9
}
np.save(rnn_defs.PROJ_DIR + rnn_defs.RESULTS_FOLDER + 'pca_%s_%s' % (seed, sim_number), dict)

# %%
