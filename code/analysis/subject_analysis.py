import bct
import numpy as np

from utils import stats, cpl_func, make_dir, pickle_dump

from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('pickle')

import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

direc = os.path.abspath('../../data/preprocessed_data/')
subjects_path = os.path.join(direc, 'subjects')
make_dir(subjects_path)

NNULLS = 100

struct_den_scale125 = np.load('../../data/original_data/Lausanne/'
                              'struct/struct_den_scale125.npy')
cortical125 = np.loadtxt(os.path.abspath('../../data/original_data/Lausanne/'
                                         'cortical/cortical125.txt'))
cor_idx125 = [i for i, val in enumerate(cortical125) if val == 1]
cor_struct_den_scale125 = struct_den_scale125[cor_idx125][:, cor_idx125]

subj_list = list(range(cor_struct_den_scale125.shape[2]))
subj_list.pop(8) #removing subject 8 because it is disconnected

#calculate minimum number of edges across subjects
min_edges = np.min([np.sum(np.triu(cor_struct_den_scale125[:, :, subj] > 0, 
                                   k=1))
                    for subj in subj_list])

for subj in subj_list:

    print('subject {}'.format(subj))
    SCmat = cor_struct_den_scale125[:, :, subj].copy()

    u, v = np.triu(SCmat, k=1).nonzero() #upper triangle indices
    wts = np.triu(SCmat, k=1)[(u, v)] #upper triangle weights
    #number of edges to remove to match the lowest density
    n_cut_edges = len(wts) - min_edges

    if n_cut_edges > 0:
        idx = np.argsort(wts) #indices that sort the weights
        SCmat = np.triu(SCmat, k=1) #upper triangle
        #remove lowest weights
        SCmat[(u[idx][:n_cut_edges], v[idx][:n_cut_edges])] = 0
        SCmat = SCmat + SCmat.T #make symmetric

    #check if disconnected
    if bct.number_of_components(SCmat) > 1:
        print('disconnected')
        continue

    conn_dict = {}

    colours = []
    cpl = []
    mean_clustering = []

    colours.append('original')
    cpl.append(cpl_func(SCmat))
    #weight conversion between 0 and 1 to compute clustering coefficient
    conv_SCmat = bct.weight_conversion(SCmat, 'normalize')
    SCmat_clustering = bct.clustering_coef_wu(conv_SCmat)
    mean_clustering.append(np.mean(SCmat_clustering))

    SCmat_strengths = np.sum(SCmat, axis = 1)
    time = {'ms': [], 'str': [], 'sa': []}
    mse = {'ms': [], 'str': [], 'sa': []}
    strengths = {'original': SCmat_strengths, 'ms': [], 'str': [], 'sa': []}

    stats_arr = list(zip(*Parallel(n_jobs = 75)(delayed(stats)(SCmat, SCmat_strengths, 'participant', seed, analysis = 'participants') for seed in range(NNULLS))))
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

    pickle_dump('/subjects/'
                'subj{}_100_nulls_preprocessed_data_dict'.format(subj),
                conn_dict)
