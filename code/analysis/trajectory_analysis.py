import numpy as np
from trajectories_utils import strength_preserving_rand_sa_trajectory

from utils import pickle_dump

from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('pickle')

import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

NNULLS = 100

SCmat = np.load('../../data/original_data/consensusSC_125_wei.npy')
    
sa_stats_arr = zip(*Parallel(n_jobs = 25)(delayed(strength_preserving_rand_sa_trajectory)(SCmat, seed = seed) for seed in range(NNULLS)))
strengths_trajectory, energy_trajectory, \
cpl_trajectory, clustering_trajectory = sa_stats_arr
    
pickle_dump('L125_strengths_trajectory_sa', strengths_trajectory)
pickle_dump('L125_energy_trajectory_sa', energy_trajectory)
pickle_dump('L125_cpl_trajectory_sa', cpl_trajectory)
pickle_dump('L125_clustering_trajectory_sa', clustering_trajectory)

SCmat = np.load('../../data/original_data/consensusSC_HCP_Schaefer400_wei.npy')
    
sa_stats_arr = zip(*Parallel(n_jobs = 25)(delayed(strength_preserving_rand_sa_trajectory)(SCmat, seed = seed) for seed in range(NNULLS)))
strengths_trajectory, energy_trajectory, \
cpl_trajectory, clustering_trajectory = sa_stats_arr
    
pickle_dump('HCP400_strengths_trajectory_sa', strengths_trajectory)
pickle_dump('HCP400_energy_trajectory_sa', energy_trajectory)
pickle_dump('HCP400_cpl_trajectory_sa', cpl_trajectory)
pickle_dump('HCP400_clustering_trajectory_sa', clustering_trajectory)