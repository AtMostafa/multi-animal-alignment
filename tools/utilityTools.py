import matplotlib.pyplot as plt
import numpy as np
import os

def get_colors(n, colormap='plasma'):
    colors = []
    cmap = plt.cm.get_cmap(colormap)
    for colorVal in np.linspace(0, 1, n+1):
        colors.append(cmap(colorVal))
    return colors[:-1]

def find_file(path, extension=['.raw.kwd']):
    """
    This function finds all the file types specified by 'extension' (ex: *.dat) in the 'path' directory
    and all its subdirectories and their sub-subdirectories etc., 
    and returns a list of all file paths
    'extension' is a list of desired file extensions: ['.dat','.prm']
    """
    if type(extension) is str:
        extension=extension.split()   #turning extension into a list with a single element
    return [os.path.join(walking[0],goodfile) for walking in list(os.walk(path)) 
         for goodfile in walking[2] for ext in extension if goodfile.endswith(ext)]