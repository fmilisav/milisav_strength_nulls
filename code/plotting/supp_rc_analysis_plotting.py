import os

import bct
import numpy as np
from scipy.stats import spearmanr

from utils import pickle_load, make_dir, phi_stats, \
                  med_avg_weight, med_euc_dist, \
                  mannwhitneyu_print, save_plot

from rich_feeder_peripheral import rich_feeder_peripheral

import seaborn as sns
import matplotlib.pyplot as plt
from nilearn.plotting import plot_connectome

from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('pickle')

import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

#loading preprocessed data

euc_distL125 = np.load('../../data/original_data/euc_distL125.npy')
euc_distL500 = np.load('../../data/original_data/euc_distL500.npy')
euc_distHCP400 = np.load('../../data/original_data/euc_distHCP400.npy')
euc_distHCP800 = np.load('../../data/original_data/euc_distHCP800.npy')

euc_dist = {'L125': euc_distL125, 'L500': euc_distL500,
            'HCP400': euc_distHCP400, 'HCP800': euc_distHCP800}

L125 = np.load('../../data/original_data/'
               'consensusSC_125_wei.npy')
L500 = np.load('../../data/original_data/'
               'consensusSC_500_wei.npy')
HCP400 = np.load('../../data/original_data/'
                 'consensusSC_HCP_Schaefer400_wei.npy')
HCP800 = np.load('../../data/original_data/'
                 'consensusSC_HCP_Schaefer800_wei.npy')

conns = {'L125': L125, 'HCP400': HCP400, 'L500': L500, 'HCP800': HCP800}

L125_prepro_data = pickle_load('../../data/preprocessed_data/'
                               'Lausanne125_preprocessed_data_dict')
L500_prepro_data = pickle_load('../../data/preprocessed_data/'
                               'Lausanne500_preprocessed_data_dict')
HCP400_prepro_data = pickle_load('../../data/preprocessed_data/'
                                 'HCP_Schaefer400_preprocessed_data_dict')
HCP800_prepro_data = pickle_load('../../data/preprocessed_data/'
                                 'HCP_Schaefer800_preprocessed_data_dict')

prepro_data = {'L125': L125_prepro_data,
               'L500': L500_prepro_data,
               'HCP400': HCP400_prepro_data,
               'HCP800': HCP800_prepro_data}

NNULLS = 100

direc = os.path.abspath('../../figures')
rc_path = os.path.join(direc, 'supp_rc_analysis')
make_dir(rc_path)

#plotting style parameters
sns.set_style("ticks")
sns.set(context=None, style=None, palette=None, font_scale=5, color_codes=None)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 20})
plt.rcParams['legend.fontsize'] = 15

for conn_key, SCmat in conns.items():
    print(conn_key)

    norm_phi_path = '{}_norm_phi.svg'.format(conn_key)
    norm_phi_abs_path = os.path.join(rc_path, norm_phi_path)

    #binary SC matrix
    SCmat_bin = SCmat.copy()
    SCmat_bin[SCmat_bin > 0] = 1

    #upper triangular mask
    mask = np.triu(np.ones(len(SCmat)), 1) > 0

    #calculating sum of rich link weights and number of rich links (rc_n)
    rfp, pvals, rc_n = rich_feeder_peripheral(SCmat, SCmat_bin, stat = 'sum')
    #sorted weights
    sort_weights = np.sort(SCmat[mask])
    #calculating denominator (sum of rc_n strongest weights)
    denom = np.zeros(rc_n.shape)
    for degthresh in range(rc_n.shape[1]):
        denom[:, degthresh] = np.sum(sort_weights[int(-rc_n[:, degthresh]):])
    #calculating rich-club ratio
    og_phi = rfp[0]/denom

    #loading null sum of rich link weights
    data = prepro_data[conn_key]['rfp']
    data_ms = np.array(data['ms'])[:NNULLS, 0, :]
    data_str = np.array(data['str'])[:NNULLS, 0, :]
    data_sa = np.array(data['sa'])[:NNULLS, 0, :]

    #calculating null maximal possible weighted connectedness
    node_degree = bct.degrees_und(SCmat)
    k = np.max(node_degree).astype(np.int64) + 1
    B_ms_list, _ = zip(*Parallel(n_jobs = 3)(delayed(bct.randmio_und_connected)(SCmat, 10, seed = seed) for seed in range(NNULLS)))
    denom = np.zeros(data_ms.shape)
    for degthresh in range(k):
        hub_idx = np.where(node_degree >= degthresh)
        null = 0
        for B_ms in B_ms_list:
            rich_conns = B_ms[np.ix_(hub_idx[0], hub_idx[0])]
            rc_n = len(np.where(rich_conns > 0)[0])
            denom[null, degthresh] = np.sum(sort_weights[int(-rc_n):])
            null += 1

    #calculating null rich-club ratios
    data_ms = data_ms/denom
    data_str = data_str/denom
    data_sa = data_sa/denom

    #calculating mean null rich-club coefficients and
    #identifying significant threshold degree values
    mean_null_phi_ms, k_sig_ms = phi_stats(og_phi, data_ms)
    mean_null_phi_str, k_sig_str = phi_stats(og_phi, data_str)
    mean_null_phi_sa, k_sig_sa = phi_stats(og_phi, data_sa)

    #indices of significant threshold degree values
    k_sig_ms_idx = np.where(k_sig_ms)[0]
    k_sig_str_idx = np.where(k_sig_str)[0]
    k_sig_sa_idx = np.where(k_sig_sa)[0]

    #normalizing rich-club ratios
    norm_phi_ms = og_phi[0]/mean_null_phi_ms
    norm_phi_str = og_phi[0]/mean_null_phi_str
    norm_phi_sa = og_phi[0]/mean_null_phi_sa

    #assessing significance of differences between norm_phi_sa and
    #norm_phi_ms/norm_phi_str
    nans = np.logical_or(np.isnan(norm_phi_sa), np.isnan(norm_phi_ms))
    x = norm_phi_sa[~nans]
    y = norm_phi_ms[~nans]
    mannwhitneyu_print(x, y, 'sa', 'ms')

    nans = np.logical_or(np.isnan(norm_phi_sa), np.isnan(norm_phi_str))
    x = norm_phi_sa[~nans]
    y = norm_phi_str[~nans]
    mannwhitneyu_print(x, y, 'sa', 'str')

    #difference between normalized rich-club ratios
    #obtained using simulated annealing and Maslov-Sneppen rewiring
    norm_phi_diff = norm_phi_sa - norm_phi_ms

    #plotting normalized rich-club ratios
    x = list(range(og_phi.shape[1]))*3
    y = list(norm_phi_ms) + list(norm_phi_str) + list(norm_phi_sa)
    hue = ['Maslov-Sneppen']*og_phi.shape[1] + \
          ['Rubinov-Sporns']*og_phi.shape[1] + \
          ['simulated annealing']*og_phi.shape[1]
    fig, ax = plt.subplots(figsize = (9, 6))
    sns.lineplot(x = x, y = y, hue = hue,
                 palette = ['dimgrey', '#00A1A1', '#2A7DBC'], ax = ax)
    x = np.append(np.append(np.array(range(og_phi.shape[1]))[k_sig_ms_idx],
                            np.array(range(og_phi.shape[1]))[k_sig_str_idx]),
                  np.array(range(og_phi.shape[1]))[k_sig_sa_idx])
    y = np.append(np.append(np.array(norm_phi_ms)[k_sig_ms_idx],
                            np.array(norm_phi_str)[k_sig_str_idx]),
                  np.array(norm_phi_sa)[k_sig_sa_idx])
    hue = ['Maslov-Sneppen']*np.sum(k_sig_ms) + \
          ['Rubinov-Sporns']*np.sum(k_sig_str) + \
          ['simulated annealing']*np.sum(k_sig_sa)
    palette = []
    if 'Maslov-Sneppen' in hue:
        palette.append('dimgrey')
    if 'Rubinov-Sporns' in hue:
        palette.append('#00A1A1')
    if 'simulated annealing' in hue:
        palette.append('#2A7DBC')
    sns.scatterplot(x = x, y = y, hue = hue, palette = palette,
                    alpha = 0.5, legend = False)
    ax.legend(frameon = False)
    ax.set(xlabel = 'degree', ylabel = 'normalized rich club ratio')
    save_plot(ax, norm_phi_abs_path)

    #plotting median average weights
    med_avg_weight_path = '{}_med_avg_weight.svg'.format(conn_key)
    med_avg_weight_abs_path = os.path.join(rc_path, med_avg_weight_path)
    med_avg_weight_arr = med_avg_weight(SCmat, SCmat_bin)
    nans = np.isnan(norm_phi_diff)
    x = med_avg_weight_arr[~nans]
    y = norm_phi_diff[~nans]
    fig, ax = plt.subplots(figsize = (9, 6))
    ax = sns.lineplot(x = list(range(len(x))), y = x, color = "#78B7C5",
                      legend = False, ax = ax)
    ax.set(xlabel = 'degree', ylabel = 'median average weight')
    save_plot(ax, med_avg_weight_abs_path)

    #plotting normalized rich-club ratio difference
    norm_phi_diff_path = '{}_norm_phi_diff.svg'.format(conn_key)
    norm_phi_diff_abs_path = os.path.join(rc_path, norm_phi_diff_path)
    fig, ax = plt.subplots(figsize = (9, 6))
    ax = sns.lineplot(x = list(range(len(x))), y = y, color = "#78B7C5",
                      legend = False, ax = ax)
    ax.set(xlabel = 'degree', ylabel = 'normalized rich club ratio difference')
    save_plot(ax, norm_phi_diff_abs_path)

    #plotting median average weight x normalized rich-club ratio difference
    med_avg_weight_x_norm_phi_diff_path = '{}_med_avg_weight_x_norm_phi_diff' \
                                          '.svg'.format(conn_key)
    med_avg_weight_x_norm_phi_diff_abs_path = os.path.join(rc_path,
                                           med_avg_weight_x_norm_phi_diff_path)
    fig, ax = plt.subplots(figsize = (9, 6))
    ax = sns.scatterplot(x = x, y = y, color = "#78B7C5", ax = ax)
    ax.set(xlabel = 'median average weight',
           ylabel = 'normalized rich club ratio difference')
    save_plot(ax, med_avg_weight_x_norm_phi_diff_abs_path)
    print('Spearman\'s rho '
          '(median average weight x normalized rich-club ratio difference): \n'
          '{}'.format(spearmanr(x, y)))

    #plotting median euclidean distance
    med_euc_dist_path = '{}_med_euc_dist.svg'.format(conn_key)
    med_euc_dist_abs_path = os.path.join(rc_path, med_euc_dist_path)
    med_euc_dist_val = med_euc_dist(SCmat_bin, euc_dist[conn_key])
    x = med_euc_dist_val[~nans]
    y = med_avg_weight_arr[~nans]
    fig, ax = plt.subplots(figsize = (9, 6))
    ax = sns.lineplot(x = list(range(len(x))), y = x, color = "#78B7C5",
                      legend = False, ax = ax)
    ax.set(xlabel = 'degree', ylabel = 'median euclidean distance')
    save_plot(ax, med_euc_dist_abs_path)

    #plotting median euclidean distance x median average weight
    med_euc_dist_x_med_avg_weight_path = '{}_med_euc_dist_x_med_avg_weight' \
                                         '.svg'.format(conn_key)
    med_euc_dist_x_med_avg_weight_abs_path = os.path.join(rc_path,
                                            med_euc_dist_x_med_avg_weight_path)
    fig, ax = plt.subplots(figsize = (9, 6))
    ax = sns.scatterplot(x = x, y = y, color = "#78B7C5", ax = ax)
    ax.set(xlabel = 'median euclidean distance',
           ylabel = 'median average weight')
    save_plot(ax, med_euc_dist_x_med_avg_weight_abs_path)
    print('Spearman\'s rho '
          '(median euclidean distance x median average weight): \n'
          '{}'.format(spearmanr(x, y)))