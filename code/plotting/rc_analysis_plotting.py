import os

import numpy as np
from scipy.stats import spearmanr

from utils import pickle_load, make_dir, phi_stats, \
                  med_avg_weight, med_euc_dist, \
                  mannwhitneyu_print, save_plot

from rich_feeder_peripheral import rich_feeder_peripheral

import seaborn as sns
import matplotlib.pyplot as plt
from nilearn.plotting import plot_connectome

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

NNULLS = 10000

direc = os.path.abspath('../../figures')
rc_path = os.path.join(direc, 'rc_analysis')
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

    #calculating sum of rich link weights
    rfp, pvals, rc_n = rich_feeder_peripheral(SCmat, SCmat_bin, stat = 'sum')
    og_phi = rfp[0]
    og_phi = og_phi[np.newaxis, :]

    #loading null sum of rich link weights
    data = prepro_data[conn_key]['rfp']
    data_ms = np.array(data['ms'])[:, 0, :]
    data_str = np.array(data['str'])[:, 0, :]
    data_sa = np.array(data['sa'])[:, 0, :]

    #calculating mean null rich-club coefficients and
    #identifying significant threshold degree values
    mean_null_phi_ms, k_sig_ms = phi_stats(og_phi, data_ms)
    mean_null_phi_str, k_sig_str = phi_stats(og_phi, data_str)
    mean_null_phi_sa, k_sig_sa = phi_stats(og_phi, data_sa)

    #indices of significant threshold degree values
    k_sig_ms_idx = np.where(k_sig_ms)[0]
    k_sig_str_idx = np.where(k_sig_str)[0]
    k_sig_sa_idx = np.where(k_sig_sa)[0]

    #normalizing rich-club coefficients
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

    #plotting rich-club
    if conn_key == 'HCP800':
        rc_conn_path = 'rc_conn_ortho_sa.svg'
        rc_abs_path = os.path.join(rc_path, rc_conn_path)

        coords = np.load('../../data/original_data/'
                         'coordsHCP800.npy')
        degree = np.sum(SCmat_bin, axis = 1)
        strength = np.sum(SCmat, axis = 1)

        idx = np.where(degree >= k_sig_sa_idx[-2])[0]
        print('k = {}'.format(k_sig_sa_idx[-2]))
        #hub-hub connections mask
        rich_mask = np.zeros((len(degree), len(degree)))
        rich_mask[np.ix_(idx, idx)] = 1
        rich_conn = SCmat.copy()
        rich_conn[rich_mask == 0] = 0

        node_size = 0.05*strength
        node_color = ['#2A7DBC']*len(degree)
        node_color = np.array(node_color)
        node_color[idx] = '#b22222'

        fig, ax = plt.subplots(figsize = (5, 5))
        plot_connectome(rich_conn, coords,
						node_color = node_color, node_size = node_size,
						edge_cmap = 'OrRd',
                        edge_vmin = np.min(rich_conn[rich_mask == 1]),
                        edge_vmax = np.max(rich_conn[rich_mask == 1]),
                        annotate = False, alpha = 0,
						axes = ax, edge_kwargs = {'lw': 1}, colorbar = True)
        save_plot(ax, rc_abs_path)

        node_size[idx] = node_size[idx]*2
        rc_conn_path = 'rc_ortho_sa.svg'
        rc_abs_path = os.path.join(rc_path, rc_conn_path)
        fig, ax = plt.subplots(figsize = (5, 5))
        plot_connectome(rich_conn, coords, edge_threshold = 1,
						node_color = node_color, node_size = node_size,
                        annotate = False, alpha = 0, axes = ax)
        save_plot(ax, rc_abs_path)

        node_size = 0.05*strength
        rc_conn_path = 'rc_ortho_ms.svg'
        rc_abs_path = os.path.join(rc_path, rc_conn_path)
        fig, ax = plt.subplots(figsize = (5, 5))
        plot_connectome(rich_conn, coords, edge_threshold = 1,
						node_color = 'dimgrey', node_size = node_size,
                        annotate = False, alpha = 0, axes = ax)
        save_plot(ax, rc_abs_path)

        rc_conn_path = 'rc_ortho_str.svg'
        rc_abs_path = os.path.join(rc_path, rc_conn_path)
        fig, ax = plt.subplots(figsize = (5, 5))
        plot_connectome(rich_conn, coords, edge_threshold = 1,
						node_color = '#00A1A1', node_size = node_size,
                        annotate = False, alpha = 0, axes = ax)
        save_plot(ax, rc_abs_path)


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