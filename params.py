import pathlib
import matplotlib
import numpy as np

# Global params
from monkey import defs as monkey_defs
from mouse import defs as mouse_defs

root = pathlib.Path("/data")
repoPath = pathlib.Path.cwd()
figPath = repoPath / 'figures'

rng = np.random.default_rng(np.random.SeedSequence(12345))
n_iter = 100
Behav_corr_TH = 0.5

def set_rc_params(dictArg:dict ={}):
    matplotlib.rcParams['backend']   = 'PDF'
    matplotlib.rcParams['xtick.major.pad'] = 1
    matplotlib.rcParams['ytick.major.pad'] = 1
    matplotlib.rcParams['axes.labelpad']   = 2
    matplotlib.rcParams['axes.titlepad']   = 5
    matplotlib.rcParams['font.size']   = 8
    matplotlib.rcParams['axes.titlesize']   = 8
    matplotlib.rcParams['axes.labelsize']   = 8
    matplotlib.rcParams['xtick.labelsize']   = 6.5
    matplotlib.rcParams['ytick.labelsize']   = 6.5
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.serif'] = 'Helvetica'
    matplotlib.rcParams['legend.frameon'] = False
    matplotlib.rcParams['legend.fancybox'] = False
    matplotlib.rcParams['figure.dpi'] = 600
    
    
    for key,val in dictArg.items():
        matplotlib.rcParams[key] = val

def load_unit_depth(df, field='depthCtx'):
    """
    used for the mice datasets
    *RAW* datafils must be saved under: `root / mouse-data-raw`
    """
    from scipy.io import loadmat
    fileName = df.file[0]
    
    rawFile = root / 'mouse-data-raw' / fileName
    rawFile = rawFile.parent / (rawFile.stem[:-3] + rawFile.suffix)
    
    a = loadmat(rawFile)[field].flatten()
    return a
