#%%
from random import sample
import subprocess
import params
import pyaldata as pyal
from params import rnn_defs
from tools import simTools as st
from sklearn.decomposition import PCA
import numpy as np
import sys

#%%
# seed = rnn_defs.SEEDS1[0]
# sim_number = 1
# df = st.get_processed_pyaldata(seed, sim_number, epoch_fun = rnn_defs.prep_exec_epoch, subtract_mean=False)
# df = pyal.dim_reduce(df, PCA(rnn_defs.n_components), 'both_rates', 'both_pca')
# df = pyal.subtract_cross_condition_mean(df)
# df = df.groupby('target_id').head(rnn_defs.BATCH_SIZE/rnn_defs.n_targets)
# pca = np.concatenate(df['both_pca'].values, axis=0)
# np.save(rnn_defs.PROJ_DIR + rnn_defs.RESULTS_FOLDER + 'pca_%s_%s' % (seed, sim_number), pca)

# # %%
# seed = rnn_defs.SEEDS1[0]
# sim_number = 1
# df = st.get_processed_pyaldata(seed, sim_number, epoch_fun = rnn_defs.exec_epoch, subtract_mean=False)
# df = pyal.dim_reduce(df, PCA(rnn_defs.n_components), 'both_rates', 'both_pca')
# df = pyal.subtract_cross_condition_mean(df)
# df = df.groupby('target_id').head(rnn_defs.BATCH_SIZE/rnn_defs.n_targets)
# pca = np.concatenate(df['both_pca'].values, axis=0)
# np.save(rnn_defs.PROJ_DIR + rnn_defs.RESULTS_FOLDER + 'pca_exec_%s_%s' % (seed, sim_number), pca)

# # %%
# seed = rnn_defs.SEEDS1[0]
# sim_number = 4
# df = st.get_processed_pyaldata(seed, sim_number, epoch_fun = rnn_defs.exec_epoch, subtract_mean=False)
# df = pyal.dim_reduce(df, PCA(rnn_defs.n_components), 'both_rates', 'both_pca')
# df = pyal.subtract_cross_condition_mean(df)
# df = df.groupby('target_id').head(rnn_defs.BATCH_SIZE/rnn_defs.n_targets)
# pca = np.concatenate(df['both_pca'].values, axis=0)
# np.save(rnn_defs.PROJ_DIR + rnn_defs.RESULTS_FOLDER + 'pca_exec_%s_%s' % (seed, sim_number), pca)

# # %%
# seed = rnn_defs.SEEDS1[0]
# sim_number = 12
# rnn_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on', 
#                                      rel_start=-40,rel_end = 40)
# df = st.get_processed_pyaldata(seed, sim_number, subtract_mean=False)
# move_onsets = df.idx_movement_on.values
# df = pyal.restrict_to_interval(df,epoch_fun = rnn_epoch)
# df = pyal.dim_reduce(df, PCA(rnn_defs.n_components), 'both_rates', 'both_pca')
# df = pyal.subtract_cross_condition_mean(df)
# df = df.groupby('target_id').head(rnn_defs.BATCH_SIZE/rnn_defs.n_targets)
# pca = np.concatenate(df['both_pca'].values, axis=0)
# dict={
#     'pca': pca,
#     'move_onsets': move_onsets,
# }
# #movement on based on matt
# np.save(rnn_defs.PROJ_DIR + rnn_defs.RESULTS_FOLDER + 'pca_movement_on_%s_%s' % (seed, sim_number), dict)

# # %%
# seed = rnn_defs.SEEDS1[0]
# sim_number = 12
# rnn_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on', 
#                                      rel_start=-40,rel_end = 40)
# df = st.get_processed_pyaldata(seed, sim_number, subtract_mean=False)
# move_onsets = df.idx_movement_on.values
# df = pyal.restrict_to_interval(df,epoch_fun = rnn_epoch)
# df = pyal.dim_reduce(df, PCA(rnn_defs.n_components), 'both_rates', 'both_pca')
# df = pyal.subtract_cross_condition_mean(df)
# df = df.groupby('target_id').head(rnn_defs.BATCH_SIZE/rnn_defs.n_targets)
# pca = np.concatenate(df['both_pca'].values, axis=0)
# dict={
#     'pca': pca,
#     'move_onsets': move_onsets,
# }
# #movement on based on threshold=9
# np.save(rnn_defs.PROJ_DIR + rnn_defs.RESULTS_FOLDER + 'pca_movement_on_thresh_%s_%s' % (seed, sim_number), dict)

# # %%
# seed = rnn_defs.SEEDS1[0]
# sim_number = 12
# rnn_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on', 
#                                     rel_start=-5,rel_end = 40)
# df = st.get_processed_pyaldata(seed, sim_number, subtract_mean=False)
# move_onsets = df.idx_movement_on.values
# df = pyal.restrict_to_interval(df,epoch_fun = rnn_epoch)
# df = pyal.dim_reduce(df, PCA(rnn_defs.n_components), 'both_rates', 'both_pca')
# df = pyal.subtract_cross_condition_mean(df)
# df = df.groupby('target_id').head(rnn_defs.BATCH_SIZE/rnn_defs.n_targets)
# pca = np.concatenate(df['both_pca'].values, axis=0)
# dict={
#     'pca': pca,
#     'move_onsets': move_onsets,
# }
# #movement on based on threshold=9
# np.save(rnn_defs.PROJ_DIR + rnn_defs.RESULTS_FOLDER + 'pca_movement_on_thresh_exec_%s_%s' % (seed, sim_number), dict)

# # %%
# seed = rnn_defs.SEEDS1[0]
# sim_number = 16
# rnn_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on', 
#                                      rel_start=-5,rel_end = 40)
# df = st.get_processed_pyaldata(seed, sim_number, subtract_mean=False)
# move_onsets = df.idx_movement_on.values
# df = pyal.restrict_to_interval(df,epoch_fun = rnn_epoch)
# df = pyal.dim_reduce(df, PCA(rnn_defs.n_components), 'both_rates', 'both_pca')
# df = pyal.subtract_cross_condition_mean(df)
# df = df.groupby('target_id').head(rnn_defs.BATCH_SIZE/rnn_defs.n_targets)
# pca = np.concatenate(df['both_pca'].values, axis=0)
# dict={
#     'pca': pca,
#     'move_onsets': move_onsets,
# }
# #movement on based on threshold=9
# np.save(rnn_defs.PROJ_DIR + rnn_defs.RESULTS_FOLDER + 'pca_movement_on_thresh_exec_%s_%s' % (seed, sim_number), dict)


# rnn_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on', 
#                                      rel_start=-40,rel_end = 40)
# df = st.get_processed_pyaldata(seed, sim_number, subtract_mean=False)
# move_onsets = df.idx_movement_on.values
# df = pyal.restrict_to_interval(df,epoch_fun = rnn_epoch)
# df = pyal.dim_reduce(df, PCA(rnn_defs.n_components), 'both_rates', 'both_pca')
# df = pyal.subtract_cross_condition_mean(df)
# df = df.groupby('target_id').head(rnn_defs.BATCH_SIZE/rnn_defs.n_targets)
# pca = np.concatenate(df['both_pca'].values, axis=0)
# dict={
#     'pca': pca,
#     'move_onsets': move_onsets,
# }
# #movement on based on threshold=9
# np.save(rnn_defs.PROJ_DIR + rnn_defs.RESULTS_FOLDER + 'pca_movement_on_thresh_%s_%s' % (seed, sim_number), dict)

# %%
sim_number = int(sys.argv[1])
seed = rnn_defs.SEEDS1[0]

rnn_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on', 
                                     rel_start=-5,rel_end = 40)
df = st.get_processed_pyaldata(seed, sim_number, subtract_mean=False)
move_onsets = df.idx_movement_on.values
df = pyal.restrict_to_interval(df,epoch_fun = rnn_epoch)
df = pyal.dim_reduce(df, PCA(rnn_defs.n_components), 'both_rates', 'both_pca')
df = pyal.subtract_cross_condition_mean(df)
df = df.groupby('target_id').head(rnn_defs.BATCH_SIZE/rnn_defs.n_targets)
pca = np.concatenate(df['both_pca'].values, axis=0)
dict={
    'pca': pca,
    'move_onsets': move_onsets,
}
#movement on based on threshold=9
np.save(rnn_defs.PROJ_DIR + rnn_defs.RESULTS_FOLDER + 'pca_movement_on_exec_%s_%s' % (seed, sim_number), dict)
