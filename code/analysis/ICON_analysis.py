import bct
import numpy as np

from utils import stats, ICON_cpl_func, make_dir, pickle_dump

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
#removing network 11 because it is big
net_list.remove(11)

for net in net_list:

    net += 1
    print('network {}'.format(net))
    mat = np.load('../../data/original_data/'
                  'weightedNets/weightedNet{}.npy'.format(net))

    conn_dict = {}

    colours = []
    cpl = []
    mean_clustering = []

    colours.append('original')
    cpl.append(ICON_cpl_func(mat))
    #weight conversion between 0 and 1 to compute clustering coefficient
    conv_mat = bct.weight_conversion(mat, 'normalize')
    mat_clustering = bct.clustering_coef_wu(conv_mat)
    mean_clustering.append(np.mean(mat_clustering))

    mat_strengths = np.sum(mat, axis = 1)
    time = {'ms': [], 'str': [], 'sa': []}
    mse = {'ms': [], 'str': [], 'sa': []}
    strengths = {'original': mat_strengths, 'ms': [], 'str': [], 'sa': []}

    stats_arr = list(zip(*Parallel(n_jobs = 75)(delayed(stats)(mat, mat_strengths, 'ICON', seed, analysis = 'ICON') for seed in range(NNULLS))))
    B_ms, time_ms, mse_ms, strengths_ms, cpl_ms, mean_clustering_ms, \
    rfp_ms, pvals_ms, null_phi_ms, rc_n = stats_arr[:10]
    B_str, time_str, mse_str, strengths_str, cpl_str, mean_clustering_str, \
    rfp_str, pvals_str, null_phi_str = stats_arr[10:19]
    B_sa, time_sa, mse_sa, strengths_sa, cpl_sa, mean_clustering_sa, \
    rfp_sa, pvals_sa, null_phi_sa = stats_arr[19:]

    time['ms'].extend(time_ms)
    time['str'].extend(time_str)
    time['sa'].extend(time_sa)

    mse['ms'].extend(mse_ms)
    mse['str'].extend(mse_str)
    mse['sa'].extend(mse_sa)

    strengths['ms'].extend(np.concatenate(strengths_ms))
    strengths['str'].extend(np.concatenate(strengths_str))
    strengths['sa'].extend(np.concatenate(strengths_sa))

    colours.extend(['Maslov-Sneppen']*NNULLS)
    cpl.extend(cpl_ms)
    mean_clustering.extend(mean_clustering_ms)

    colours.extend(['Rubinov-Sporns']*NNULLS)
    cpl.extend(cpl_str)
    mean_clustering.extend(mean_clustering_str)

    colours.extend(['simulated annealing']*NNULLS)
    cpl.extend(cpl_sa)
    mean_clustering.extend(mean_clustering_sa)

    conn_dict['time'] = time
    conn_dict['mse'] = mse
    conn_dict['strengths'] = strengths
    conn_dict['colours'] = colours
    conn_dict['cpl'] = cpl
    conn_dict['mean_clustering'] = mean_clustering

    pickle_dump('/weightedNets/'
                'net{}_100_nulls_preprocessed_data_dict'.format(net),
                conn_dict)