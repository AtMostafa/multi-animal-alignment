# Preserved neural dynamics across animals performing similar behaviour

This repository includes code to reproduce the figures in [Mostafa & Chang et. al (bioRxiv, 2022)](https://www.biorxiv.org/content/10.1101/2022.09.26.509498v1).

## Getting Started

Create a conda environment with ```conda env create -f env.yml``` and activate the environment with ```conda activate cca```.

Note that a version of MATLAB is needed for matlabengine, which is used for Fig S2. Comment out the matlabengine line if you do not have MATLAB installed. Otherwise, change the version of matlabengine according to the MATLAB version you have installed. 

## Reproducing figures

TODO: add data

Each figure in the paper has an associated Jupyter notebook under ```paper/```. Running the cells reproduces all of the subfigures. 

For the RNN simulations associated with Figure 5 and Figure S10, first run the simulations:
1. ```cd rnn```
2. ```bash run.sh```

## System Requirements
The code has been tested on Linux (Ubuntu 18.04). 

## License
[MIT](https://opensource.org/license/mit/)
