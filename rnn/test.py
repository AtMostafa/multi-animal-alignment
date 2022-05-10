#%%

from config_manager import base_configuration
from rnn.simulation.config_template import ConfigTemplate
from params import rnn_defs
from rnn.simulation.runner import Runner
import os
import contextlib
import numpy as np
import torch

    
def test_model(config_dir):
    """ 
    Test a trained model

    Parameters
    ----------
    config_dir: str
        directory for configuration file

    """
    #setup runner
    config_path = config_dir + "config.yaml"
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        config = base_configuration.BaseConfiguration(
            configuration= config_path, template = ConfigTemplate.base_config_template)

    #set random seeds 
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    runner = Runner(config, rnn_defs.PROJ_DIR, training = False)

    #test
    datadir, output, activity1 = runner.run_test()

    return datadir, output, activity1

# %%
