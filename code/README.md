# `code`

## `analysis`

The [`analysis`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/analysis) folder contains the code used to conduct the main analyses and generate the various benchmarking and network statistics used for downstream statistical analyses and [`plotting`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/plotting). The resulting preprocessed data files are available in [`preprocessed_data`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data). All the scripts used for benchmarking and network analysis are labelled `analysis`. Below are short descriptions of the support files.

- [`rich_feeder_peripheral.py`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/analysis/rich_feeder_peripheral.py) contains a function for computing weighted rich-club statistics. It was adapted from a [netneurotools function](https://netneurotools.readthedocs.io/en/latest/generated/netneurotools.metrics.rich_feeder_peripheral.html#netneurotools.metrics.rich_feeder_peripheral) written by Justine Hansen.

- [`strength_preserving_rand_rs.py`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/analysis/strength_preserving_rand_rs.py) contains an implementation of the Rubinov-Sporns (2011) strength-preserving randomization algorithm. It was adapted from a function intended for signed networks written in MATLAB by Mika Rubinov for the [Brain Connectivity Toolbox](https://sites.google.com/site/bctnet).

- [`strength_preserving_rand_sa.py`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/analysis/strength_preserving_rand_sa.py) contains the main function for generating strength-preserving randomized networks using simulated annealing as presented in the manuscript. It should only take a couple seconds to run on the provided structural consensus networks. For a detailed assessment of the time-performance tradeoff of the simulated annealing procedure, please refer to Fig. S12 of the manuscript.

- [`strength_preserving_rand_sa_dir.py`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/analysis/strength_preserving_rand_sa_dir.py) contains the simulated annealing algorithm for strength-preserving randomization adapted to directed networks.

- [`strength_preserving_rand_sa_energy_thresh.py`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/analysis/strength_preserving_rand_sa_energy_thresh.py) contains the simulated annealing algorithm for strength-preserving randomization adapted to include an energy threshold.

- [`strength_preserving_rand_sa_flexE.py`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/analysis/strength_preserving_rand_sa_flexE.py) contains the simulated annealing algorithm for strength-preserving randomization adapted to allow for flexible user-specified energy functions (at the cost of a slower execution time).

- [`trajectories_utils.py`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/analysis/trajectories_utils.py) contains the simulated annealing algorithm for strength-preserving randomization adapted to track energy and global network features.

- [`utils.py`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/analysis/utils.py) contains generic auxiliary functions used in analysis scripts.

- [`struct_consensus.py`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/analysis/struct_consensus.py) contains a function for generating a distance-dependent group consensus structural connectivity matrix. It was adapted from a [netneurotools function](https://netneurotools.readthedocs.io/en/latest/generated/netneurotools.networks.struct_consensus.html) to allow for user-specified numbers of intra/inter-hemispheric links.

- [`consensus_SC_L125_custom_densities.py`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/analysis/consensus_SC_L125_custom_densities.py) is a script for generating group consensus structural connectivity matrices at custom densities from the 219 cortical nodes resolution of the Lausanne dataset. [`consensus_SCs_fixed_nedges.py`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/analysis/consensus_SCs_fixed_nedges.py) is a script for generating group consensus structural connectivity matrices with a fixed number of edges from multiple resolutions of the Lausanne dataset (L060: 114 nodes, L125: 219 nodes, L250: 448 nodes, L500: 1000 nodes). The resulting preprocessed data files are available in [`scaling_analysis`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data/scaling_analysis).


## `plotting`

The [`plotting`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/plotting) folder contains the code used to run the statistical tests and generate the figures presented in the manuscript.
Scripts respect the naming convention established in [`analysis`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/analysis).


### Running the analyses

1. Git clone this repository.
2. Download the necessary preprocessed data files from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10729405.svg)](https://doi.org/10.5281/zenodo.10729405) and place them in the appropriate [`preprocessed_data`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data) folder.
3. Install the relevant Python packages by building and activating a conda environment from one of the two provided .yml files (a different file is provided for the [`analysis`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/analysis) and the [`plotting`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/plotting) scripts). To do so, in the command line, type:

```bash
cd milisav_strength_nulls
conda env create -f code/plotting/milisav_str_nulls_plotting.yml
conda activate milisav_str_nulls_plotting
```

or replacing `code/plotting/milisav_str_nulls_plotting.yml` by `code/analysis/milisav_str_nulls_analysis.yml` and `milisav_str_nulls_plotting` by `milisav_str_nulls_analysis` as appropriate. This should take a couple minutes.

4. To reproduce the manuscript figures, simply type:

```bash
python code/plotting/plotting.py
```

replacing `plotting.py` by the plotting script of your choice.
