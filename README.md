# Preserved neural dynamics across animals performing similar behaviour

By: Mostafa Safaie<sup>\*</sup>, Joanna C. Chang<sup>\*</sup>, Junchol Park, Lee E. Miller, Joshua T. Dudman, Matthew G. Perich and Juan A. Gallego.

This repository includes code to reproduce the figures in [Safaie & Chang et. al., Nature, 2023](https://www.nature.com/articles/s41586-023-06714-0).

## Getting Started

Create a conda environment with ```conda env create -f env.yml``` and activate the environment with ```conda activate cca```.

Note that a version of MATLAB is needed for matlabengine, which is used for Fig S2.
Comment out the matlabengine line if you do not have MATLAB installed.
Otherwise, change the version of matlabengine according to the MATLAB version you have installed.

## Reproducing Figures

Each figure in the paper has an associated Jupyter notebook under [*/paper*](/paper).
Running the cells reproduces all of the panels.

For the RNN simulations associated with Figure 5 and Figure S10, first run the simulations:

```bash
cd rnn && bash rnn.sh
```

## System Requirements

The code has been tested on Linux (Ubuntu >18.04).

## Data Availability

After the publication of the paper, all the monkey datasets used in this work (and more) were deposited in [this](https://dandiarchive.org/dandiset/000688) DANDI repository, in the NWB format.
Following an easy format conversion (e.g., see [here](https://www.nwb.org/how-to-use/)), it will be easy to reproduce the analyses on the monkey datasets.
The mouse datasets will be made available on reasonable request.

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

## Funders

This work was supported in part by: 
- grant number H2020-MSCA-IF-2020-101025630 from the Commission of the European Union (M.S.)
- grant number 108908/Z/15/Z from the Wellcome Trust (J.C.C.)
-  grant numbers NS053603 and NS074044 from the NIH National Institute of Neurological Disorders and Stroke (L.E.M.)
-  grant _chercheurs-boursiers en intelligence artificielle_ from the Fonds de recherche du Quebec Santé (M.G.P.)
-  grant number EP/T020970/1 from the UKRI Engineering and Physical Sciences Research Council (J.A.G.)
-  grant number ERC-2020-StG-949660 from the European Research Council (J.A.G.).
