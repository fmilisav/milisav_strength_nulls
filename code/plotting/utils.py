import os
import pickle

import bct
import numpy as np
import pingouin as pg
from scipy.stats import spearmanr, mannwhitneyu, iqr

import seaborn as sns
import matplotlib.pyplot as plt

from strength_preserving_rand_sa import strength_preserving_rand_sa


#general functions

def pickle_load(file):
    return pickle.load(open(file + '.pickle', "rb"))

def make_dir(path):
    try: os.mkdir(path)
    except OSError as error:
        print(error)


#plotting functions

#plotting parameters
marginal_kws = {'element': 'step', 'alpha': 0.7}
scatter_kws = {'rasterized':True}

#function to plot strengths in empirical and randomized networks

def null_color(null):

    if null == 'ms':
        color = 'dimgrey'
    elif null == 'str':
        color = '#00A1A1'
    else: color = '#2A7DBC'

    return color

def plot_morphospace(x, y, hue, palette, morpho_path, linewidth = None,
                     plot_legend = False, legend_title = None):

    ax = sns.jointplot(x = x, y = y, hue = hue, palette = palette, 
                       rasterized = True, linewidth = linewidth)
    ax.ax_joint.set(xlabel = 'Characteristic path length', 
                    ylabel = 'Clustering')
    if plot_legend:
        ax.ax_joint.legend(title = legend_title, loc='center left', 
                           bbox_to_anchor=(1.15, 0.5), frameon = False)
    else:
        ax.ax_joint.legend().remove()
    ax.savefig(morpho_path, dpi = 300)
    plt.close(ax.fig)

def plot_strengths_regplots(og_strengths, rewired_strengths, 
							nnulls, color, regplot_abs_path):

	x = np.tile(og_strengths, nnulls)
	y = rewired_strengths
	SCmat_n = len(og_strengths)
	corrs = []
	for i in range(0, nnulls*SCmat_n, SCmat_n):
		curr_x = x[i: i + SCmat_n]
		curr_y = y[i: i + SCmat_n]
		empirical = spearmanr(curr_x, curr_y)[0]
		corrs.append(empirical)

	r = np.mean(corrs)
	sd = np.std(corrs, ddof = 1)
	label = "\N{GREEK SMALL LETTER RHO} {} {} {} {}".format(u"\u2248",
															round(r, 3),
															u"\u00B1",
															round(sd, 3))
	line_kws = {'label':label}
	g = sns.jointplot(x = x, y = y, kind = 'reg', color = color,
					  height = 10, ratio = 7, marginal_kws = marginal_kws,
					  scatter_kws = scatter_kws, line_kws = line_kws,
					  seed = 0)
	g.ax_joint.set(xlabel = 'strengths in empirical', 
				   ylabel = 'strengths in randomized')
	g.ax_joint.legend()
	g.ax_joint.legend(frameon = False)
	xlim = g.ax_joint.get_xlim()
	ylim = g.ax_joint.get_ylim()
	lims = [np.min([xlim, ylim]),
			np.max([xlim, ylim])]
	g.ax_joint.plot(lims, lims, 'k-', zorder = 0)
	set_joint_plot_lims(g, lims)
	g.savefig(regplot_abs_path, dpi = 300)
	plt.close(g.fig)

	return np.array(corrs), r, sd

def set_joint_plot_lims(g, lims):

    g.ax_joint.set_xlim(lims)
    g.ax_joint.set_ylim(lims)
    g.ax_joint.set_aspect('equal')

#https://stackoverflow.com/a/26456731
def adjust_yaxis(ax, ydif, v):

    inv = ax.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, ydif))
    miny, maxy = ax.get_ylim()
    miny, maxy = miny - v, maxy - v
    if -miny > maxy or (-miny == maxy and dy > 0):
        nminy = miny
        nmaxy = miny * (maxy + dy)/(miny + dy)
    else:
        nmaxy = maxy
        nminy = maxy * (miny + dy)/(maxy + dy)
    ax.set_ylim(nminy + v, nmaxy + v)

def align_yaxis(ax1, v1, ax2, v2):

    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    adjust_yaxis(ax2,(y1 - y2)/2, v2)
    adjust_yaxis(ax1,(y2 - y1)/2, v1)

def save_plot(ax, path, close = True):

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', dpi = 300)
    if close:
        plt.close(fig)


#statistical functions

#function to normalize connectome weights
def scale(values, vmin, vmax, axis=None):

    s = (values - values.min(axis=axis)) / \
        (values.max(axis=axis) - values.min(axis=axis))
    s = s * (vmax - vmin)
    s = s + vmin

    return s

#function to calculate hub-related statistics
def hub_stats(strengths):

    strengths_len = len(strengths)
    strengths_iqr = iqr(strengths)
    strengths_q3 = np.percentile(strengths, 75)

    hub_threshold = strengths_q3 + 3*strengths_iqr
    hubs = np.where(strengths > hub_threshold)[0]
    n_hubs = len(hubs)

    return strengths_len, hub_threshold, hubs, n_hubs

#function to calculate characteristic path length
#with a negative log transform from weight to length
def cpl_func(A):

    A_len = -np.log(A)
    #shortest path length matrix
    A_dist = bct.distance_wei(A_len)[0]
    np.fill_diagonal(A_dist, np.nan)
    cpl = np.nanmean(A_dist)

    return cpl

#function to perform Mann-Whitney U rank test
#and print results
def mannwhitneyu_print(x, y, xlabel, ylabel):
	u, p_val = mannwhitneyu(x, y)
	print('{} median={}, IQR={}'.format(xlabel, np.median(x), iqr(x)))
	print('{} median={}, IQR={}'.format(ylabel, np.median(y), iqr(y)))
	print('Mann-Whitney U rank test {}: u={}, p={}'.format(xlabel + '-' + 
                                                           ylabel, u, p_val))
	print('cles = {}'.format(pg.compute_effsize(x, y, eftype = 'CLES')))

#function to generate null networks using simulated annealing
#for different numbers of iterations and calculate strengths
def niter_null_strengths(SCmat, seed):

    B_sa, _ = strength_preserving_rand_sa(SCmat, seed = seed, niter = 1000)
    strengths_sa_1000_iter = np.sum(B_sa, axis = 1)

    B_sa, _ = strength_preserving_rand_sa(SCmat, seed = seed, niter = 100000)
    strengths_sa_100000_iter = np.sum(B_sa, axis = 1)

    return strengths_sa_1000_iter, strengths_sa_100000_iter

#function to calculate mean null rich-club coefficients and 
#identify significant threshold degree values
def phi_stats(og_phi, null_phi):

    mean_null_phi = []
    k_sig = []
    for i in range(og_phi.shape[1]):
        null_distrib = null_phi[:, i]
        null_distrib_mean = np.mean(null_distrib)
        mean_null_phi.append(null_distrib_mean)
        demeaned_null_distrib = null_distrib - null_distrib_mean
        demeaned_phi = og_phi[:, i] - null_distrib_mean
        #p-value derived as the proportion of the null distribution
        #that is more extreme than the empirical value
        p_sum = (np.abs(demeaned_null_distrib) >= np.abs(demeaned_phi)).sum()
        p = p_sum/len(demeaned_null_distrib)
        #Bonferroni correction
        if p < 0.05/og_phi.shape[1]:
            k_sig.append(1)
        else:
            k_sig.append(0)

    return np.array(mean_null_phi), k_sig

#function to calculate median average weight
#at different threshold degrees
def med_avg_weight(SCmat, SCmat_bin):

    degree = np.sum(SCmat_bin, axis = 1)
    strength = np.sum(SCmat, axis = 1)
    avg_weight = strength/degree

    k = np.max(degree).astype(np.int64) + 1
    med_avg_weight = []
    for degthresh in range(k):
        idx = np.where(degree >= degthresh)[0]
        med_avg_weight.append(np.median(avg_weight[idx]))

    return np.array(med_avg_weight)

#function to calculate median inter-hub euclidean distance
#at different threshold degrees
def med_euc_dist(SCmat_bin, euc_dist):

    degree = np.sum(SCmat_bin, axis = 1)
    conn_euc_dist = euc_dist.copy()
    #connections mask
    conn_euc_dist[SCmat_bin == 0] = 0

    med_euc_dist = []
    k = np.max(degree).astype(np.int64) + 1
    for degthresh in range(k):
        idx = np.where(degree >= degthresh)[0]
        #hub-hub connections mask
        rich_mask = np.zeros((len(degree), len(degree)))
        rich_mask[np.ix_(idx, idx)] = 1
        conn_euc_dist[rich_mask == 0] = 0
        conn_euc_dist[conn_euc_dist == 0] = np.nan
        med_euc_dist.append(np.nanmedian(conn_euc_dist))

    return np.array(med_euc_dist)

#function to build a dictionary of morphospace features
def build_morpho_dict(path, net_list):

    features = ['cpl', 'mean_clustering', 'colours']

    features_dict = {}
    features_dict['net'] = []
    for feature in features:
        features_dict[feature] = []

    for net in net_list:
        file = '{}_100_nulls_preprocessed_data_dict'.format(net)
        data = pickle_load(path + file)
        net_len = len(data['colours'])
        for feature in features:
            features_dict[feature].extend(data[feature])
            
        features_dict['net'].extend([net]*net_len)
    features_dict['net'] = np.array(features_dict['net'])
    for feature in features:
        features_dict[feature] = np.array(features_dict[feature])

    return features_dict
