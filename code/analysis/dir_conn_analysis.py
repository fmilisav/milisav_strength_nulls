import numpy as np

from utils import dir_stats, pickle_dump

from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('pickle')

import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

NNULLS = 10000

drosophila = np.load('../../data/original_data/drosophila.npy')
mouse = np.load('../../data/original_data/mouse.npy')
rat = np.load('../../data/original_data/rat.npy')
macaque = np.load('../../data/original_data/macaque.npy')

dir_conns = {'drosophila': drosophila, 'mouse': mouse, 
			 'rat': rat, 'macaque': macaque}

dir_conn_prepro_data_dict = {}
for animal, SCmat in dir_conns.items():

	dir_conn_prepro_data_dict[animal] = {}
	dir_conn_prepro_data_dict[animal]['mse'] = []
	dir_conn_prepro_data_dict[animal]['time'] = []
	dir_conn_prepro_data_dict[animal]['strengths'] = {'in': [], 'out': []}
        
	_, mse_sa, time, strengths_in, strengths_out = list(zip(*Parallel(n_jobs = 75)(delayed(dir_stats)(SCmat, seed) for seed in range(NNULLS))))

	dir_conn_prepro_data_dict[animal]['mse'].extend(mse_sa)
	dir_conn_prepro_data_dict[animal]['time'].extend(time)
	dir_conn_prepro_data_dict[animal]['strengths']['in'].extend(np.concatenate(strengths_in))
	dir_conn_prepro_data_dict[animal]['strengths']['out'].extend(np.concatenate(strengths_out))

pickle_dump('dir_conn_prepro_data_dict', dir_conn_prepro_data_dict)