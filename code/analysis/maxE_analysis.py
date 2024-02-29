import numpy as np
from utils import max_stats, pickle_dump

from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('pickle')

import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

NNULLS = 100

conns = {}

for res in ['125', '500']:
    conns['L{}'.format(res)] = np.load('../../data/original_data/'
                                       'consensusSC_{}_wei.npy'.format(res))

for res in ['400', '800']:
    conns['HCP{}'.format(res)] = np.load('../../data/original_data/'
                                         'consensusSC_HCP_Schaefer{}'
                                         '_wei.npy'.format(res))

for conn_key, SCmat in conns.items():
    strengths_sa = Parallel(n_jobs = 25)(delayed(max_stats)(SCmat, seed) for seed in range(NNULLS))
    strengths_sa = np.concatenate(strengths_sa)

    pickle_dump('{}_maxE_strengths_sa'.format(conn_key), strengths_sa)