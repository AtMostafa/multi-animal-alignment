# Preserved neural dynamics across animals performing similar behaviour

By: Mostafa Safaie<sup>\*</sup>, Joanna C. Chang<sup>\*</sup>, Junchol Park, Lee E. Miller, Joshua T. Dudman, Matthew G. Perich and Juan A. Gallego.

This repository includes code to reproduce the figures in [Safaie & Chang et. al., bioRxiv, 2022](https://www.biorxiv.org/content/10.1101/2022.09.26.509498v1).

## Getting Started

Create a conda environment with ```conda env create -f env.yml``` and activate the environment with ```conda activate cca```.

Note that a version of MATLAB is needed for matlabengine, which is used for Fig S2.
Comment out the matlabengine line if you do not have MATLAB installed.
Otherwise, change the version of matlabengine according to the MATLAB version you have installed.

## Reproducing Figures

Each figure in the paper has an associated Jupyter notebook under [*/paper*](/paper).
Running the cells reproduces all of the panels.

For the RNN simulations associated with Figure 5 and Figure S10, first run the simulations:
1. ```cd rnn```
2. ```bash run.sh```

## System Requirements

The code has been tested on Linux (Ubuntu >18.04).

## License

[MIT](https://opensource.org/license/mit/).

## Code Credit

Code from the following sources have been copied and used here under  [*/packages*](/packages) or [*/tools*](/tools) with slight modifications:

* [config_manager](https://github.com/seblee97/config_package) from [Sebastian Lee](https://github.com/seblee97)
* [pyaldata](https://github.com/NeuralAnalysis/PyalData) from [Neural Analysis](https://github.com/NeuralAnalysis)
* [dPCA](https://github.com/machenslab/dPCA/tree/master/matlab) from [machenslab](https://github.com/machenslab)
* [TME](https://github.com/gamaleldin/TME/tree/master) from [Gamaleldin Elsayed](https://github.com/gamaleldin)

Associated publications are referenced in the paper.

## Questions

Questions can be directed to the corresponding authors, as issues on this repository, or to [Mostafa](mailto:mostafa.safaie@gmail.com) or [Joanna](mailto:joanna.changc@gmail.com).
