import matplotlib

def set_rc_params(dictArg:dict ={}):
    matplotlib.rcParams['xtick.major.pad'] = 1
    matplotlib.rcParams['ytick.major.pad'] = 1
    matplotlib.rcParams['axes.labelpad']   = 2
    matplotlib.rcParams['axes.titlepad']   = 3
    matplotlib.rcParams['axes.titlesize']   = 'medium'
    matplotlib.rcParams['axes.labelsize']   = 'small'
    matplotlib.rcParams['xtick.labelsize']   = 'x-small'
    matplotlib.rcParams['ytick.labelsize']   = 'x-small'
    matplotlib.rcParams['text.usetex'] = True
    
    
    for key,val in dictArg.items():
        matplotlib.rcParams[key] = val
