import matplotlib

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
    matplotlib.rcParams['legend.frameon'] = False
    matplotlib.rcParams['legend.fancybox'] = False
    
    
    for key,val in dictArg.items():
        matplotlib.rcParams[key] = val
