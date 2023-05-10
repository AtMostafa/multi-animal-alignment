import os
import pyaldata as pyal

# DIRECTORIES ##################################################
PROJ_DIR=os.path.dirname(os.path.realpath(__file__))+'/'
RESULTS_FOLDER = "results/"
# RESULTS_FOLDER = "results_prep/"
DATA_FOLDER = "data/"
FIGURES_FOLDER = "figures/"
CONFIGS_FOLDER = "simulation/configs/"

#LOGGING
PRINT_EPOCH = 5

#DATA
MAX_Y_POS = 7
MOVEMENT_SPEED_THRESHOLD = 6.0
LOSS_THRESHOLD = 0.2
MAX_TRAINING_TRIALS = 500#1001#500
MIN_TRAINING_TRIALS = 50#1000#50

#RANDOM SEEDS
SEEDS1 = range(1000020,1000030)
SEEDS2 = range(1000030,1000040)
# SEEDS1 = range(1000020,1000030)
# SEEDS2 = range(1000030,1000040)

#TRAINING
BATCH_SIZE = 64

#DATA PROCESSING
pca_dims = 10

BIN_SIZE = .01  # sec
n_components = 10 
n_targets = 8
seed_idx_ex = 4
trial_ex = 1

WINDOW_prep = (-.40, .05)  # sec
WINDOW_exec = (-.05, .40)  # sec
WINDOW_prep_exec = (-.40, .40)  # sec
prep_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on',
                                     rel_start=int(WINDOW_prep[0]/BIN_SIZE),
                                     rel_end=int(WINDOW_prep[1]/BIN_SIZE)
                                    )
exec_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on', 
                                     rel_start=int(WINDOW_exec[0]/BIN_SIZE),
                                     rel_end=int(WINDOW_exec[1]/BIN_SIZE)
                                    )
prep_exec_epoch = pyal.generate_epoch_fun(start_point_name='idx_movement_on', 
                                     rel_start=int(WINDOW_prep_exec[0]/BIN_SIZE),
                                     rel_end=int(WINDOW_prep_exec[1]/BIN_SIZE)
                                    )