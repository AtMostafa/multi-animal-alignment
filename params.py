import pathlib
import matplotlib
import numpy as np

# Global params
root = pathlib.Path("/data")
rng = np.random.default_rng(np.random.SeedSequence(12345))

def set_rc_params(dictArg:dict ={}):
    matplotlib.rcParams['xtick.major.pad'] = 1
    matplotlib.rcParams['ytick.major.pad'] = 1
    matplotlib.rcParams['axes.labelpad']   = 2
    matplotlib.rcParams['axes.titlepad']   = 5
    matplotlib.rcParams['axes.titlesize']   = 'x-large'
    matplotlib.rcParams['axes.labelsize']   = 'large'
    matplotlib.rcParams['xtick.labelsize']   = 'medium'
    matplotlib.rcParams['ytick.labelsize']   = 'medium'
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.serif'] = 'Helvetica'
    matplotlib.rcParams['legend.frameon'] = False
    matplotlib.rcParams['legend.fancybox'] = False
    
    
    for key,val in dictArg.items():
        matplotlib.rcParams[key] = val
