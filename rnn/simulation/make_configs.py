#%%
import os
from params import rnn_defs
from config_manager import base_configuration
from rnn.simulation.config_template import ConfigTemplate
import numpy as np

#%%
MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

def create_config(config_num, property_name = None, property_value = None, changes_dict = None, base_config =None):
    assert(((property_name is not None) & (property_value is not None)) or (changes_dict is not None))
    if base_config is None:
        base_config = 'config.yaml'
    config_path = os.path.join(MAIN_FILE_PATH, 'configs/' + base_config)

    configuration = base_configuration.BaseConfiguration(
        configuration= config_path, template = ConfigTemplate.base_config_template)

    if changes_dict is None:
        configuration.amend_property(property_name=property_name, new_property_value=property_value)
    else:
        for property_name in changes_dict.keys():
            property_value = changes_dict[property_name]
            configuration.amend_property(property_name=property_name, new_property_value=property_value)
    
    save_dir = rnn_defs.PROJ_DIR + rnn_defs.CONFIGS_FOLDER 
    file_name = 'config_' + str(config_num) + '.yaml'
    configuration.save_configuration(folder_path=save_dir, file_name = file_name)


#%%
n_simulation = 19

for i, gin in enumerate([0.1,0.01]):
    for j, noise in enumerate([0.1,0.2]):
        for k, p_recurrent in enumerate([1.0,0.1]):
            for l, reg in enumerate([True, False]):
                if reg:
                    changes_dict = {
                        'gin': gin,
                        'noise': noise,
                        'p_recurrent': p_recurrent,
                    }
                else:
                    changes_dict = {
                        'alpha1': 0.0, 
                        'gamma1': 0.0, 
                        'beta1': 0.0, 
                        'gin': gin,
                        'noise': noise,
                        'p_recurrent': p_recurrent,
                    }
                create_config(n_simulation,changes_dict = changes_dict)
                n_simulation += 1
# %%
