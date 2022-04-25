#%%
from config_manager import base_configuration
from rnn.simulation.config_template import ConfigTemplate
from params import rnn_defs
from rnn.simulation.runner import Runner
import os
import contextlib

def set_random_seed(rand_seed):
    import numpy as np
    import torch

    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

def setup_runner(config_dir, datafile = None, noise = None):
    config_path = config_dir + "config.yaml"
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        config = base_configuration.BaseConfiguration(
            configuration= config_path, template = ConfigTemplate.base_config_template)
        
    if datafile is not None:
        data_folder = rnn_defs.DATA_FOLDER
        datadir = data_folder + str(config.seed) + '/' + datafile
        config.amend_property(property_name="datadir", new_property_value=datadir)
    if noise is not None:
        config.amend_property(property_name="noise", new_property_value=noise)
    
    set_random_seed(config.seed)

    runner = Runner(config, rnn_defs.PROJ_DIR, training = False)

    return runner
    
def test_model(config_dir, epoch = None, datafile = None, noise = None):
    """
    Test a trained model.

    Parameters
    ----------
    config_dir: str
        directory where model is stored
    epoch: int
        epoch model was at during training
    datafile: str
        data to test the model on
    r1_input: np array (trials x tsteps x neurons)
            pre-defined activity for the first RNN
    r2_input: np array (trials x tsteps x neurons)
        pre-defined activity for the second RNN
    noise: double
        noise to include in network

    """
    runner = setup_runner(config_dir, datafile, noise)
    datadir, output, activity1 = runner.run_test(epoch)
    return datadir, output, activity1

# %%
