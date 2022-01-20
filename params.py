import pathlib
import string
import matplotlib
import numpy as np

# Global params
from monkey import defs as monkey_defs
from mouse import defs as mouse_defs

root = pathlib.Path("/data")
repoPath = pathlib.Path.cwd()
figPath = repoPath / 'figures'
LargeFig = (7,8)
MedFig = (4.3,4.3)

rng = np.random.default_rng(np.random.SeedSequence(12345))
n_iter = 100
Behav_corr_TH = 0.5


def set_rc_params(dictArg:dict ={}):
    matplotlib.rcParams['backend']   = 'PDF'
    matplotlib.rcParams['lines.markersize'] = 4
    matplotlib.rcParams['xtick.major.pad']  = 1
    matplotlib.rcParams['ytick.major.pad']  = 1
    matplotlib.rcParams['axes.labelpad']    = 2
    matplotlib.rcParams['axes.titlepad']    = 5
    matplotlib.rcParams['font.size']        = 8
    matplotlib.rcParams['axes.titlesize']   = 8
    matplotlib.rcParams['axes.labelsize']   = 8
    matplotlib.rcParams['xtick.labelsize']  = 6.5
    matplotlib.rcParams['ytick.labelsize']  = 6.5
    matplotlib.rcParams['legend.fontsize']  = 6.5
    matplotlib.rcParams['legend.title_fontsize']   = 6.5
    matplotlib.rcParams['text.usetex']      = True
    matplotlib.rcParams['font.family']      = 'sans-serif'
    matplotlib.rcParams['font.serif']       = 'Helvetica'
    matplotlib.rcParams['legend.frameon']   = False
    matplotlib.rcParams['legend.fancybox']  = False
    matplotlib.rcParams['figure.dpi']       = 600
    
    
    for key,val in dictArg.items():
        matplotlib.rcParams[key] = val


def add_panel_caption(axes: tuple, offsetX: tuple, offsetY: tuple, **kwargs):
    """
    This function adds letter captions (a,b,c,d) to Axes in axes
    at top left, with the specified offset, in RELATIVE figure coordinates
    """
    assert len(axes)==len(offsetX)==len(offsetY), 'Bad input!'

    fig=axes[0].get_figure()
    fbox=fig.bbox
    for ax,dx,dy,s in zip(axes,offsetX,offsetY,string.ascii_uppercase):
        axbox=ax.get_window_extent()
        try:
            ax.text(x=(axbox.x0/fbox.xmax)-abs(dx), y=(axbox.y1/fbox.ymax)+abs(dy),
                    s=s, fontweight='extra bold', fontsize=10, ha='left', va='center',
                    transform=fig.transFigure,**kwargs)
        except: #to cover 3D axes
            ax.text2D(x=(axbox.x0/fbox.xmax)-abs(dx), y=(axbox.y1/fbox.ymax)+abs(dy),
                    s=s, fontweight='extra bold', fontsize=10, ha='left', va='center',
                    transform=fig.transFigure,**kwargs)
            


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

class panels:
    "sizes of different panels in the paper"
    TinyH = 0.5
    SmallH = 1.2
    MedH = SmallH + TinyH
    BigH = 3
    
    schmatic = (2,MedH)
    raster = (2.4,SmallH)
    velocity = (LargeFig[0]-schmatic[0],TinyH)
    proj_3d_align = (LargeFig[0],SmallH)
    cca = (1.5, SmallH)
    cca_hist = (2,SmallH)
    decoding_hist = (2,SmallH)
    neuro_behav_corr = (2.1, SmallH)
    
class colors:
    "colors for different data types in the paper"
    MainCC = 'k'
    LowerCC = 'gray'
    UpperCC = 'cornflowerblue'
    MonkeyPts = 'xkcd:brown'
    MousePts = 'xkcd:violet'
    UniversalDecoder = 'goldenrod'
    LeftTrial = (0,1,0,1)
    RightTrial = (1,0,0,1)
