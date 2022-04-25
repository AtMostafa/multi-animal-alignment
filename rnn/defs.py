import os
import pyaldata as pyal

# DIRECTORIES ##################################################
PROJ_DIR=os.path.dirname(os.path.realpath(__file__))+'/'
RESULTS_FOLDER = "results/"
DATA_FOLDER = "data/"
FIGURES_FOLDER = "figures/"
CONFIGS_FOLDER = "simulation/configs/"

#LOGGING
PRINT_EPOCH = 5

#DATA
MAX_Y_POS = 10
MOVEMENT_SPEED_THRESHOLD = 6.0
LOSS_THRESHOLD = 0.3
MAX_TRAINING_TRIALS = 500

#RANDOM SEEDS
SEEDS1 = [1000010,1000011, 1000012, 1000013, 1000014, 1000015, 1000016, 1000017, 1000018, 1000019]
SEEDS2 = [1000020,1000021, 1000022, 1000023, 1000024, 1000025, 1000026, 1000027, 1000028, 1000029]
BATCH_SIZE = 64

#DATA PROCESSING
START_POINT = 'idx_movement_on'
REL_START = -45
REL_END = 45
pca_dims = 10

BIN_SIZE = .01  # sec
n_components = 10 
n_targets = 8
seed_idx_ex = 3
trial_ex = 15

# WINDOW_prep = (-.40, .05)  # sec
# WINDOW_exec = (-.05, .40)  # sec
# prep_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on',
#                                      rel_start=int(WINDOW_prep[0]/BIN_SIZE),
#                                      rel_end=int(WINDOW_prep[1]/BIN_SIZE)
#                                     )
# exec_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on', 
#                                      rel_start=int(WINDOW_exec[0]/BIN_SIZE),
#                                      rel_end=int(WINDOW_exec[1]/BIN_SIZE)
#                                     )

WINDOW_prep = (-.45, .0)  # sec
WINDOW_exec = (-.0, .45)  # sec
WINDOW_prep_exec = (-.45, .45)  # sec

prep_epoch = pyal.generate_epoch_fun(start_point_name='idx_go_cue',
                                     rel_start=int(WINDOW_prep[0]/BIN_SIZE),
                                     rel_end=int(WINDOW_prep[1]/BIN_SIZE)
                                    )
exec_epoch = pyal.generate_epoch_fun(start_point_name='idx_go_cue', 
                                     rel_start=int(WINDOW_exec[0]/BIN_SIZE),
                                     rel_end=int(WINDOW_exec[1]/BIN_SIZE)
                                    )
prep_exec_epoch = pyal.generate_epoch_fun(start_point_name='idx_go_cue', 
                                     rel_start=int(WINDOW_prep_exec[0]/BIN_SIZE),
                                     rel_end=int(WINDOW_prep_exec[1]/BIN_SIZE)
                                    )
