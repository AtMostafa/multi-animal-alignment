# %%

import numpy as np
from torch.utils.data import Dataset
import torch

class Task_Params(Dataset):
    """ Task parameters"""
    def __init__(self, datadir):
        params = np.load(datadir+'.npy',allow_pickle = True).item()['params']

        self.output_dim = params['output_dim']
        self.tsteps = params['tsteps'] 
        self.input_dim = params['input_dim']
        self.dt = params['dt'] 
        self.use_velocities = params['use_velocities']

class Task_Dataset(Dataset):
    """Centerout reach task dataset."""

    def __init__(self, datadir, training = True):
        """
        Parameters
        ----------
        datadir: string
            Path to the .npy file with data file
        training: boolean
            whether you are training the network
        """
        info = np.load(datadir+'.npy',allow_pickle = True).item()

        if training:
            self.data = info
        else:
            self.data = info['test_set1']
        self.output_dim = info['params']['output_dim']
        self.stimulus = torch.from_numpy(self.data['stimulus'])
        self.target = torch.from_numpy(self.data['target'][:,:,:self.output_dim])
        self.labels = torch.from_numpy(self.data['target_param'])
        
    def __len__(self):
        return self.stimulus.shape[0]

    def __getitem__(self, idx):

        stim_idx = self.stimulus[idx]
        target_idx = self.target[idx]

        return stim_idx, target_idx

    def get_stimulus(self):
        return self.stimulus
    
    def get_go_onset(self):
        return self.go_onset
    
    def get_target(self):
        return self.target

    def get_stimulus_target(self):
        return self.stimulus, self.target

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    datadir = '../data/dataset_chewie_bl0pos'
    data = np.load(datadir+'.npy',allow_pickle = True).item()
    print(data.keys())
    print(data['params'])

    dataset = Task_Dataset(datadir)

    datadir2 = '../data/dataset_chewie_prep'
    data2 = np.load(datadir2+'.npy',allow_pickle = True).item()
    print(data2.keys())
    print(data2['params'])
    dataset2 = Task_Dataset(datadir2)

    fig, axs = plt.subplots(nrows =2, figsize = (5,5), sharex=True)
    i = 0
    n_in = dataset.stimulus.shape[-1]
    #stimuli
    for j in range(n_in):
        axs[0].plot(dataset.stimulus[i,:,j], label= j) #hold
    # axs[0].plot(dataset.stimulus[i,:,1], label = 'cos', linestyle = '--') #cos
    # axs[0].plot(dataset.stimulus[i,:,2], label = 'sin', linestyle = 'dotted') #sin
    # axs[0].plot(dataset.stimulus[i,:,3], label = 'length', linestyle = '-.') #length
    axs[0].legend(title = 'Stimuli', bbox_to_anchor=(1.1, 1.05))

    #target: pos
    axs[1].plot(dataset.target[i,:,0], label = 'x pos') #x vel
    axs[1].plot(dataset.target[i,:,1], label = 'y pos') #y vel
    axs[1].legend(title = 'Target pos',  bbox_to_anchor=(1.1, 1.05))

    dataset = dataset2
    fig, axs = plt.subplots(nrows =2, figsize = (5,5), sharex=True)
    i = 0
    n_in = dataset.stimulus.shape[-1]
    #stimuli
    for j in range(n_in):
        axs[0].plot(dataset.stimulus[i,:,j], label= j) #hold
    # axs[0].plot(dataset.stimulus[i,:,1], label = 'cos', linestyle = '--') #cos
    # axs[0].plot(dataset.stimulus[i,:,2], label = 'sin', linestyle = 'dotted') #sin
    # axs[0].plot(dataset.stimulus[i,:,3], label = 'length', linestyle = '-.') #length
    axs[0].legend(title = 'Stimuli', bbox_to_anchor=(1.1, 1.05))

    #target: pos
    axs[1].plot(dataset.target[i,:,0], label = 'x pos') #x vel
    axs[1].plot(dataset.target[i,:,1], label = 'y pos') #y vel
    axs[1].legend(title = 'Target pos',  bbox_to_anchor=(1.1, 1.05))

# %%
