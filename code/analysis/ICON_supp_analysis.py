import bct
import numpy as np

from utils import supp_stats, make_dir, pickle_dump

from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('pickle')

import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

direc = os.path.abspath('../../data/preprocessed_data/')
nets_path = os.path.join(direc, 'weightedNets')
make_dir(nets_path)

NNULLS = 100

net_list = list(range(38))
net_list.remove(5)
net_list.remove(11)
net_list.remove(34)

for net in net_list:

    net += 1
    print('network {}'.format(net))
    mat = np.load('../../data/original_data/'
                  'weightedNets/weightedNet{}.npy'.format(net))

    conn_dict = {}

    P_mat = {}
    consensus = {}
    Q = {}
    zrand = {}

    modularity = []
    assortativity = []

    supp_stats_arr = list(zip(*Parallel(n_jobs = 50)(delayed(supp_stats)(mat, seed) for seed in range(NNULLS))))
    B_ms, consensus_ms, Q_ms, zrand_ms, assortativity_ms = supp_stats_arr[:5]
    B_sa, consensus_sa, Q_sa, zrand_sa, assortativity_sa = supp_stats_arr[5:10]

    P_mat['ms'] = np.mean(np.array(B_ms), axis = 0)
    P_mat['sa'] = np.mean(np.array(B_sa), axis = 0)
    consensus['ms'] = consensus_ms
    consensus['sa'] = consensus_sa
    zrand['ms'] = zrand_ms
    zrand['sa'] = zrand_sa

    modularity.extend(np.mean(np.array(Q_ms), axis = 1))
    assortativity.extend(assortativity_ms)

    modularity.extend(np.mean(np.array(Q_sa), axis = 1))
    assortativity.extend(assortativity_sa)

    conn_dict['P_mat'] = P_mat
    conn_dict['consensus'] = consensus
    conn_dict['zrand'] = zrand
    conn_dict['modularity'] = modularity
    conn_dict['assortativity'] = assortativity

    pickle_dump('/weightedNets/'
                'net{}_100_nulls_preprocessed_data_dict_supp'.format(net),
                conn_dict)