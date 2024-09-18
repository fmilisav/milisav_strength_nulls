import numpy as np

from utils import in_sa_dir_stats, pickle_dump

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

    dir_conn_prepro_data_dict = {}
    dir_conn_prepro_data_dict['time'] = []
    dir_conn_prepro_data_dict['mse'] = []
    dir_conn_prepro_data_dict['strengths'] = {'in': [], 'out': []}

    stats_arr = list(zip(*Parallel(n_jobs = 50)(delayed(in_sa_dir_stats)(SCmat, seed) for seed in range(NNULLS))))
    _, time, mse, strengths_in, strengths_out = stats_arr

    dir_conn_prepro_data_dict['time'].extend(time)
    dir_conn_prepro_data_dict['mse'].extend(mse)

    dir_conn_prepro_data_dict['strengths']['in'].extend(np.concatenate(strengths_in))
    dir_conn_prepro_data_dict['strengths']['out'].extend(np.concatenate(strengths_out))

    pickle_dump('in_strengths_sa_{}_prepro_data_dict'.format(animal), dir_conn_prepro_data_dict)