#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline for running simulations.
"""
import time, os
import argparse
import datetime
from params import rnn_defs
from rnn.simulation.runner import Runner
import numpy as np

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

def get_args():
    """ Get command line arguments. """
    parser = argparse.ArgumentParser(description='Simulation parameters')
    parser.add_argument(
        'seed', 
        type=int, 
        help = "random seed"
        )
    parser.add_argument(
        'sim_number', 
        type =int,
        help = "simulation number", 
        )
    parser.add_argument(
        '-c', '--config',
        type=str,
        help="path to configuration file for simulations",
        default = 'config.yaml'
        )
    parser.add_argument(
        '-file', '--datafile',
        type=str,
        help="data file",
        )
    parser.add_argument(
        '-gpu', '--gpu_id', 
        type =int,
        help = "gpu to use", 
        )
    parser.add_argument(
        '-cca', '--ccareg', 
        help = "whether to use cca regularization, with path to calculated pcas", 
        nargs='?', const='c'
    )
    parser.add_argument(
        '-a', '--argument', 
        help = "change given argument", 
        nargs='+'
    )

    args = parser.parse_args()

    return args

def set_random_seed(rand_seed):
    """ Set random seeds in places that use random generators. """
    import numpy as np
    import torch

    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

def set_outdir(config):
    """ Set the output directory where results should be saved."""
    results_folder = rnn_defs.RESULTS_FOLDER
    seed_str = str(config.seed)

    outdir = results_folder + seed_str + '/' + str(config.sim_number) + '/' 
    os.makedirs(outdir, exist_ok=True)
    print("outdir:", outdir)

    config.amend_property(property_name="outdir", new_property_value=outdir)

    return config

def set_datadir(config):
    """ Set the directory where the data is located. """
    data_folder = rnn_defs.DATA_FOLDER
    datadir = data_folder + config.datafile
    print('datadir:', datadir)
    config.amend_property(property_name="datadir", new_property_value=datadir)

    return config

def set_sim_metadata(config, args):
    """ Set metadata specific to simulation. """
    config = set_outdir(config)
    config = set_datadir(config)

    raw_datetime = datetime.datetime.fromtimestamp(time.time())
    exp_timestamp = raw_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    config.amend_property('timestamp', exp_timestamp)

    return config

def get_config(args):
    """ Get and edit the configuration object which specifies parameters for the simulation."""
    from config_manager import base_configuration
    from rnn.simulation.config_template import ConfigTemplate

    config_path = os.path.join(MAIN_FILE_PATH, 'configs/', args.config)
    configuration = base_configuration.BaseConfiguration(
        configuration= config_path, template = ConfigTemplate.base_config_template)
    
    configuration.amend_property(property_name="seed", new_property_value=args.seed)
    configuration.amend_property(property_name="sim_number", new_property_value=args.sim_number)
    
    if args.ccareg:
        configuration.amend_property(property_name="ccareg", new_property_value=True)
        configuration.amend_property(property_name="pcas_file", new_property_value=args.ccareg)
    if args.argument:
        print(args.argument)
        assert (len(args.argument)%2 == 0)
        for _ in range(len(args.argument)//2):
            name = args.argument.pop(0)
            value = args.argument.pop(0)
            print(name, value)
            configuration.amend_property(property_name=name, new_property_value=int(value)) #TODO: delete this
    if args.gpu_id:
        configuration.amend_property(property_name="gpu_id", new_property_value=args.gpu_id)
#         print("gpu", args.gpu_id)
    if args.datafile:
        configuration.amend_property(property_name="datafile", new_property_value=args.datafile)

    return configuration

def run(config):
    """ Set up and train simulation model. """
    runner = Runner(config, os.getcwd())
    runner.run_train()
    return runner

if __name__ == "__main__":
    import time

    starttime = time.time()

    args = get_args()
    set_random_seed(args.seed)
    config = get_config(args)
    config = set_sim_metadata(config, args)
    runner = run(config)

    from tools.simTools import graph_position
    import matplotlib.pyplot as plt
    _, output, _ = runner.run_test(model_loaded = True)

    # check training
    plt.figure()
    graph_position(output)
    plt.savefig(config.outdir + "output.png")

    endtime = time.time()
    print('Total time: %.2f'%(endtime-starttime))
