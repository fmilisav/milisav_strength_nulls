# `data`

## `original_data`

The [`original_data`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/original_data) folder contains the original data used for benchmarking:

- [`consensusSC_125_wei.npy`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/original_data/consensusSC_125_wei.npy) and [`consensusSC_500_wei.npy`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/original_data/consensusSC_500_wei.npy) contain the structural consensus networks derived from the 219 and 1000 cortical nodes resolution of the Lausanne dataset (Griffa et al., 2019), respectively. Individual participant data is openly available [HERE](https://doi.org/10.5281/zenodo.2872624).

- [`consensusSC_HCP_Schaefer400_wei.npy`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/original_data/consensusSC_HCP_Schaefer400_wei.npy) and [`consensusSC_HCP_Schaefer800_wei.npy`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/original_data/consensusSC_HCP_Schaefer800_wei.npy) contain the structural consensus networks derived from the HCP dataset (Van Essen et al., 2013; Park et al., 2021) using the Schaefer parcellation (Schaefer et al, 2018) at the 400 and 800 cortical nodes resolution, respectively.

- [`coordsHCP800.npy`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/original_data/coordsHCP800.npy) contains the coordinates of the nodes in the Schaefer800 parcellation (Schaefer et al, 2018).

- [`euc_distL125.npy`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/original_data/euc_distL125.npy) and [`euc_distL500.npy`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/original_data/euc_distL500.npy) contain the Euclidean distances between nodes in the Cammoun parcellation (Cammoun et al., 2012) at the 219 and 1000 cortical nodes resolution, respectively.

- [`euc_distHCP400.npy`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/original_data/euc_distHCP400.npy) and [`euc_distHCP800.npy`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/original_data/euc_distHCP800.npy) contain the Euclidean distances between nodes in the Schaefer parcellation (Schaefer et al., 2018) at the 400 and 800 cortical nodes resolution, respectively.

- [`drosophila.npy`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/original_data/drosophila.npy) contains the drosophila directed structural connectivity matrix (Chiang et al., 2011; Shih et al., 2015; Worrell et al., 2017).

- [`macaque.npy`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/original_data/macaque.npy) contains the macaque directed structural connectivity matrix (Scholtens et al., 2014; Stephan et al., 2001).

- [`mouse.npy`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/original_data/mouse.npy) contains the mouse directed structural connectivity matrix (Oh et al., 2014; Rubinov, 2015).

- [`rat.npy`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/original_data/rat.npy) contains the rat directed structural connectivity matrix (Bota et al., 2015).

If you use this data, make sure to cite the relevant publications. See the *Methods* section of the [paper](https://www.biorxiv.org/content/10.1101/2024.02.23.581792v1) for more information. 


## `preprocessed_data`

The [`preprocessed_data`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data) folder contains the preprocessed data data used for downstream statistical analyses and [`plotting`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/code/plotting):

- `Lausanne125_preprocessed_data_dict.pickle`, `Lausanne500_preprocessed_data_dict.pickle`, `HCP_Schaefer400_preprocessed_data_dict.pickle`, and `HCP_Schaefer800_preprocessed_data_dict.pickle` are dictionaries containing benchmarking and network statistics for the low-resolution (219 nodes) Lausanne network, the high-resolution (1000 nodes) Lausanne network, the low-resolution (400 nodes) HCP network, and the high-resolution (800 nodes) HCP network, respectively. Data includes randomization execution times, performances in terms of MSE, null strengths, characteristic path lengths, mean clustering coefficients, and weighted rich-club statistics.

- `dir_conn_prepro_data_dict.pickle` is a dictionary containing randomization execution times, performances in terms of MSE, and null in- and out-strengths from the directed animal connectomes.

- [`L125_maxE_strengths_sa.pickle`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data/L125_maxE_strengths_sa.pickle), [`L500_maxE_strengths_sa.pickle`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data/L500_maxE_strengths_sa.pickle), [`HCP400_maxE_strengths_sa.pickle`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data/HCP400_maxE_strengths_sa.pickle), and [`HCP800_maxE_strengths_sa.pickle`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data/HCP800_maxE_strengths_sa.pickle) contain null strengths obtained using simulated anealing with a maximum absolute error energy function for the low-resolution (219 nodes) Lausanne network, the high-resolution (1000 nodes) Lausanne network, the low-resolution (400 nodes) HCP network, and the high-resolution (800 nodes) HCP network, respectively.

- [`Lausanne125_niter_preprocessed_data_dict.pickle`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data/Lausanne125_niter_preprocessed_data_dict.pickle) is a dictionary containing randomization execution times, performances in terms of MSE, and number of iterations per annealing stage for the low-resolution (219 nodes) Lausanne network.

- [`L125_strengths_trajectory_sa.pickle`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data/L125_strengths_trajectory_sa.pickle) and [`HCP400_strengths_trajectory_sa.pickle`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data/HCP400_strengths_trajectory_sa.pickle) contain null strengths obtained throughout the annealing procedure for the low-resolution (219 nodes) Lausanne network and the low-resolution (400 nodes) HCP network, respectively.

- [`L125_energy_trajectory_sa.pickle`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data/L125_energy_trajectory_sa.pickle) and [`HCP400_energy_trajectory_sa.pickle`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data/HCP400_energy_trajectory_sa.pickle) contain performances in terms of MSE obtained throughout the annealing procedure for the low-resolution (219 nodes) Lausanne network and the low-resolution (400 nodes) HCP network, respectively.

- [`L125_cpl_trajectory_sa.pickle`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data/L125_cpl_trajectory_sa.pickle) and [`HCP400_cpl_trajectory_sa.pickle`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data/HCP400_cpl_trajectory_sa.pickle) contain characteristic path lengths obtained throughout the annealing procedure for the low-resolution (219 nodes) Lausanne network and the low-resolution (400 nodes) HCP network, respectively.

- [`L125_clustering_trajectory_sa.pickle`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data/L125_clustering_trajectory_sa.pickle) and [`HCP400_clustering_trajectory_sa.pickle`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data/HCP400_clustering_trajectory_sa.pickle) contain mean clustering coefficients obtained throughout the annealing procedure for the low-resolution (219 nodes) Lausanne network and the low-resolution (400 nodes) HCP network, respectively.

- The [`scaling_analysis`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data/scaling_analysis) folder contains group consensus structural connectivity matrices generated at custom densities from the 219 cortical nodes resolution of the Lausanne dataset, as well as dictionaries containing benchmarking statistics for these networks. Data includes randomization execution times and performances in terms of MSE. File names contain the number of links in the network. 

- The [`subjects`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data/subjects) folder contains dictionaries of benchmarking and network statistics obtained for individual participant connectomes from the 219 cortical nodes resolution of the Lausanne dataset. Data includes randomization execution times, performances in terms of MSE, null strengths, characteristic path lengths, and mean clustering coefficients. Individual participant connectomes are openly available [HERE](https://doi.org/10.5281/zenodo.2872624).

- The [`weightedNets`](https://github.com/fmilisav/milisav_strength_nulls/blob/main/data/preprocessed_data/weightedNets) folder contains dictionaries of benchmarking and network statistics obtained for real-world complex weighted networks. Data includes randomization execution times, performances in terms of MSE, null strengths, characteristic path lengths, and mean clustering coefficients. Individual networks are openly available [HERE](https://figshare.com/s/22c5b72b574351d03edf).
