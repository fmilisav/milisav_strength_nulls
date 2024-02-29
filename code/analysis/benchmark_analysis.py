import numpy as np
from utils import stats, pickle_dump

from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('pickle')

import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

NNULLS = 10000

conns = {}

for res in ['125', '500']:
    conns['Lausanne{}'.format(res)] = np.load('../../data/original_data/'
                                      'consensusSC_{}_wei.npy'.format(res))

for res in ['400', '800']:
    conns['HCP_Schaefer{}'.format(res)] = np.load('../../data/original_data/'
                                                  'consensusSC_HCP_Schaefer{}'
                                                  '_wei.npy'.format(res))

for conn_key, SCmat in conns.items():

    conn_dict = {}

    colours = []
    cpl = []
    mean_clustering = []
    rc_n_list = []

    P_mat = {}
    time = {'ms': [], 'str': [], 'sa': []}
    mse = {'ms': [], 'str': [], 'sa': []}
    strengths = {'ms': [], 'str': [], 'sa': []}
    null_phi = {'ms': [], 'str': [], 'sa': []}
    rfp = {'ms': [], 'str': [], 'sa': []}
    pvals = {'ms': [], 'str': [], 'sa': []}

    SCmat_strengths = np.sum(SCmat, axis = 1)
    stats_arr = list(zip(*Parallel(n_jobs = 75)(delayed(stats)(SCmat, SCmat_strengths, conn_key, seed) for seed in range(NNULLS))))
    B_ms, time_ms, mse_ms, strengths_ms, cpl_ms, mean_clustering_ms, \
    rfp_ms, pvals_ms, null_phi_ms, rc_n = stats_arr[:10]
    B_str, time_str, mse_str, strengths_str, cpl_str, mean_clustering_str, \
    rfp_str, pvals_str, null_phi_str = stats_arr[10:19]
    B_sa, time_sa, mse_sa, strengths_sa, cpl_sa, mean_clustering_sa, \
    rfp_sa, pvals_sa, null_phi_sa = stats_arr[19:]

    P_mat['ms'] = np.mean(np.array(B_ms), axis = 0)
    P_mat['str'] = np.mean(np.array(B_str), axis = 0)
    P_mat['sa'] = np.mean(np.array(B_sa), axis = 0)

    time['ms'].extend(time_ms)
    time['str'].extend(time_str)
    time['sa'].extend(time_sa)

    mse['ms'].extend(mse_ms)
    mse['str'].extend(mse_str)
    mse['sa'].extend(mse_sa)

    strengths['ms'].extend(np.concatenate(strengths_ms))
    strengths['str'].extend(np.concatenate(strengths_str))
    strengths['sa'].extend(np.concatenate(strengths_sa))

    null_phi['ms'].extend(np.concatenate(null_phi_ms))
    null_phi['str'].extend(np.concatenate(null_phi_str))
    null_phi['sa'].extend(np.concatenate(null_phi_sa))

    rfp['ms'].extend(rfp_ms)
    rfp['str'].extend(rfp_str)
    rfp['sa'].extend(rfp_sa)

    pvals['ms'].extend(pvals_ms)
    pvals['str'].extend(pvals_str)
    pvals['sa'].extend(pvals_sa)

    rc_n_list.extend(rc_n)

    colours.extend(['Maslov-Sneppen']*NNULLS)
    cpl.extend(cpl_ms)
    mean_clustering.extend(mean_clustering_ms)

    colours.extend(['Rubinov-Sporns']*NNULLS)
    cpl.extend(cpl_str)
    mean_clustering.extend(mean_clustering_str)

    colours.extend(['simulated annealing']*NNULLS)
    cpl.extend(cpl_sa)
    mean_clustering.extend(mean_clustering_sa)

    conn_dict['P_mat'] = P_mat
    conn_dict['time'] = time
    conn_dict['mse'] = mse
    conn_dict['strengths'] = strengths
    conn_dict['null_phi'] = null_phi
    conn_dict['rfp'] = rfp
    conn_dict['pvals'] = pvals
    conn_dict['rc_n'] = rc_n_list
    conn_dict['colours'] = colours
    conn_dict['cpl'] = cpl
    conn_dict['mean_clustering'] = mean_clustering

    pickle_dump('{}_preprocessed_data_dict'.format(conn_key), conn_dict)
