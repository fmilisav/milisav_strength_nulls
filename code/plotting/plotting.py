import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import bct
import numpy as np
from netneurotools.modularity import zrand
from scipy.stats import kstest, linregress

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from nilearn.plotting import plot_connectome

import pandas as pd

from utils import align_yaxis, cpl_func, hub_stats, mannwhitneyu_print, \
				  make_dir, niter_null_strengths, null_color, pickle_load, \
				  plot_morphospace, plot_strengths_regplots, save_plot, scale

import pickle5 as pickle

from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('pickle')

import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

from strength_preserving_rand_rs import strength_preserving_rand_rs
from strength_preserving_rand_sa import strength_preserving_rand_sa

NNULLS = 10000

#loading preprocessed data

L125 = np.load('../../data/original_data/consensusSC_125_wei.npy')
L500 = np.load('../../data/original_data/consensusSC_500_wei.npy')
HCP400 = np.load('../../data/original_data/'
				 'consensusSC_HCP_Schaefer400_wei.npy')
HCP800 = np.load('../../data/original_data/'
				 'consensusSC_HCP_Schaefer800_wei.npy')

conn_keys = ['L125', 'L500', 'HCP400', 'HCP800']
conns = {'L125': L125, 'L500': L500, 'HCP400': HCP400, 'HCP800': HCP800}

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

strengths = {'L125': prepro_data['L125']['strengths'],
			 'L500': prepro_data['L500']['strengths'],
			 'HCP400': prepro_data['HCP400']['strengths'],
			 'HCP800': prepro_data['HCP800']['strengths']}

og_strengths_dict = {'L125': np.sum(L125, axis = 1),
					 'L500': np.sum(L500, axis = 1),
					 'HCP400': np.sum(HCP400, axis = 1),
					 'HCP800': np.sum(HCP800, axis = 1)}

L125_niter_prepro_data = pickle_load('../../data/preprocessed_data/'
									'Lausanne125_niter_preprocessed_data_dict')

drosophila = np.load('../../data/original_data/drosophila.npy')
mouse = np.load('../../data/original_data/mouse.npy', 
				allow_pickle = True)
rat = np.load('../../data/original_data/rat.npy')
macaque = np.load('../../data/original_data/macaque.npy', 
				  allow_pickle = True)

dir_conns = {'drosophila': drosophila, 'mouse': mouse, 
			 'rat': rat, 'macaque': macaque}

dir_conn_prepro_data = pickle_load('../../data/preprocessed_data/'
								   'dir_conn_prepro_data_dict')

#plotting style parameters
sns.set_style("ticks")
sns.set(context=None, style=None, palette=None, 
		font_scale=5, color_codes=None)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 30})
plt.rcParams['legend.fontsize'] = 20

direc = os.path.abspath('../../figures')

#STRENGTH SEQUENCE PRESERVATION
regplots_path = os.path.join(direc, 'str_seq')
make_dir(regplots_path)

conn_corrs_dict = {}
print('sequence statistics')
for conn_key, SCmat in conns.items():
	print(conn_key)

	conn_corrs = {}
	for null in ['ms', 'str', 'sa']:

		regplot_path = 'regplot_{}_{}.svg'.format(null, conn_key)
		regplot_abs_path = os.path.join(regplots_path, regplot_path)

		color = null_color(null)

		og_strengths = og_strengths_dict[conn_key]
		rewired_strengths = strengths[conn_key][null]

		corrs, _, _ = plot_strengths_regplots(og_strengths, rewired_strengths, 
											  NNULLS, color, regplot_abs_path)
		
		conn_corrs[null] = np.array(corrs)

	x = conn_corrs['sa']
	y = conn_corrs['str']
	mannwhitneyu_print(x, y, 'sa', 'str')

	x = conn_corrs['sa']
	y = conn_corrs['ms']
	mannwhitneyu_print(x, y, 'sa', 'ms')

	x = conn_corrs['str']
	y = conn_corrs['ms']
	mannwhitneyu_print(x, y, 'str', 'ms')

	conn_corrs_dict[conn_key] = conn_corrs

#DISTRIBUTION PRESERVATION
distrib_path = os.path.join(direc, 'str_distrib')
make_dir(distrib_path)

print('distribution statistics')
plt.rcParams.update({'font.size': 15})

for conn_key, SCmat in conns.items():
	print(conn_key)

	cdfplot_path = 'cdfplot_{}.png'.format(conn_key)
	cdfplot_abs_path = os.path.join(distrib_path, cdfplot_path)

	cdfplot_zoom = 'cdfplot_zoom_{}.png'.format(conn_key)
	cdfplot_zoom_abs_path = os.path.join(distrib_path, cdfplot_zoom)

	ksplot_path = 'ksplot_{}.svg'.format(conn_key)
	ksplot_abs_path = os.path.join(distrib_path, ksplot_path)

	og_strengths = og_strengths_dict[conn_key]
	SCmat_n = len(og_strengths)

	fig, ax = plt.subplots()
	fig2, ax2 = plt.subplots()

	KS_dict = {}
	for null in ['ms', 'str', 'sa']:

		rewired_strengths = strengths[conn_key][null]

		color = null_color(null)

		KS_arr = []
		p_arr = []
		for i in range(NNULLS):
			x = rewired_strengths[i*SCmat_n:(i + 1)*SCmat_n]
			sns.ecdfplot(x = x, color = color, legend = False, ax = ax2)
			ks, p = kstest(x, og_strengths)
			KS_arr.append(ks)
			p_arr.append(p)

		KS_dict[null] = np.array(KS_arr)

		sns.kdeplot(x = KS_arr, common_norm = True, cut = 0, 
			  		ax = ax, color = color, fill = True)
		
	ax.set_xlabel('Kolmogorov-Smirnov statistic')
	ax.set_box_aspect(1)
	save_plot(ax, ksplot_abs_path)

	sns.ecdfplot(x = og_strengths, color = "#6725A1", legend = False, ax = ax2)

	ax2.set_xlabel('strengths')
	ax2.set_box_aspect(1)
	save_plot(ax2, cdfplot_abs_path)

	if conn_key == 'L125':
		ax2.axis([0.1, 0.2, 0.8, 1.0])
	elif conn_key == 'L500':
		ax2.axis([0.1, 0.25, 0.8, 1.0])
	elif conn_key == 'HCP400':
		ax2.axis([15, 25, 0.8, 1.0])
	else:
		ax2.axis([15, 30, 0.80, 1.0])
	save_plot(ax2, cdfplot_zoom_abs_path)

	x = KS_dict['str']
	y = KS_dict['sa']
	mannwhitneyu_print(x, y, 'str', 'sa')

	x = KS_dict['ms']
	y = KS_dict['sa']
	mannwhitneyu_print(x, y, 'ms', 'sa')

	x = KS_dict['str']
	y = KS_dict['ms']
	mannwhitneyu_print(x, y, 'str', 'ms')

#hubs
print('hubs statistics')

#generating example null networks

L500_hub_examples = {}
SCmat = L500.copy()
for null_type in ['og', 'ms', 'str', 'sa']:

	if null_type == 'og':
		null = SCmat.copy()
	elif null_type == 'ms':
		null = bct.randmio_und_connected(SCmat, 10, seed = 0)[0]
	elif null_type == 'str':
		null = strength_preserving_rand_rs(SCmat,
										   R = L500_hub_examples['ms'],
										   seed = 0)
	else:
		null=strength_preserving_rand_sa(SCmat,
										 R=L500_hub_examples['ms'],
										 seed = 0)[0]

	L500_hub_examples[null_type] = null

#fetching cortical coordinates for connectome plotting

lausanne_path = '../../data/original_data/Lausanne/'
cortical = np.loadtxt(lausanne_path + 'cortical/cortical500.txt')
cor_idx = [i for i, val in enumerate(cortical) if val == 1]
coords = np.load(lausanne_path + 'coords/coords500.npy')
cor_coords = coords[cor_idx]

#plotting example null networks

for null_type in ['og', 'ms', 'str', 'sa']:

	conn_path = '{}_hub_example_ortho.png'.format(null_type)
	conn_abs_path = os.path.join(distrib_path, conn_path)

	SCmat = L500_hub_examples[null_type].copy()

	null_strengths = np.sum(SCmat, axis = 1)
	strengths_len, hub_threshold, hubs, n_hubs = hub_stats(null_strengths)

	if n_hubs > 0:

		#non-hubs
		periphery = np.where(null_strengths < hub_threshold)[0]

		hubs_frac = n_hubs/strengths_len
		hubs_frac_str = '{}% hubs'.format(np.around(hubs_frac*100,
										  decimals = 2))
		#check if heavy-tailed
		if hubs_frac > 0.009:
			title = 'heavy-tailed - ' + hubs_frac_str
		else:
			title = hubs_frac_str

		node_size = 5*null_strengths
		node_size[hubs] *= 5

		node_colors = [mcolors.to_rgb('black')]*strengths_len
		node_colors = np.array(node_colors)
		node_colors[hubs] = mcolors.to_rgb('firebrick')

		#matrix of hub connections
		hubs_matrix = SCmat.copy()
		hubs_matrix[np.ix_(periphery, periphery)] = 0

		#matrix of hub-hub connections
		rich_matrix = SCmat.copy()
		rich_matrix[periphery, :] = 0
		rich_matrix[:, periphery] = 0

		#setting all the colormap ranges to that of the empirical connectome
		if null_type == 'og':
			edge_vmin = np.min(hubs_matrix[hubs_matrix > 0])
			print('edge vmin: {}'.format(edge_vmin))
			edge_vmax = np.max(hubs_matrix)
			print('edge vmax: {}'.format(edge_vmax))

		fig, ax = plt.subplots(figsize = (5, 5))
		plot_connectome(hubs_matrix, cor_coords,
						node_color = node_colors, node_size = node_size,
						edge_cmap = 'crest', annotate = False, alpha = 0,
						axes = ax, edge_kwargs = {'lw': 0.5},
						edge_vmin = edge_vmin, edge_vmax = edge_vmax)
		plot_connectome(rich_matrix, cor_coords,
						node_color = node_colors, node_size = node_size,
						edge_cmap = 'flare', annotate = False, alpha = 0,
						axes = ax, edge_kwargs = {'lw': 0.5},
						edge_vmin = edge_vmin, edge_vmax = edge_vmax,
						title = title)
		save_plot(ax, conn_abs_path)

og_strengths = og_strengths_dict['L500']
strengths_len, hub_threshold, hubs, n_hubs = hub_stats(og_strengths)

og_hubs_frac = n_hubs/strengths_len
og_hubs_partition = np.array([0]*strengths_len)
og_hubs_partition[hubs] = 1

hubs_frac_dict = {}
rand_idx_dict = {}
for null in ['ms', 'str', 'sa']:
    print(null)

    rewired_strengths = strengths['L500'][null]

    heavy_tailed = 0
    hubs_frac_list = []
    zrand_list = []
    for i in range(0, NNULLS*strengths_len, strengths_len):
        curr_strengths = rewired_strengths[i: i + strengths_len]

        strengths_len, hub_threshold, hubs, n_hubs = hub_stats(curr_strengths)

        hubs_frac = n_hubs/strengths_len
        hubs_frac_list.append(hubs_frac)

		#check if heavy-tailed
        if hubs_frac > 0.009:
            heavy_tailed += 1

        hubs_partition = np.array([0]*strengths_len)
        hubs_partition[hubs] = 1

		#compare empirical and null hubs partitions
        zrand_list.append(zrand(og_hubs_partition, hubs_partition))

    heavy_tailed_frac = heavy_tailed/NNULLS
    hubs_frac_dict[null] = hubs_frac_list
    print('heavy-tailedness: {}%'.format(np.around(heavy_tailed_frac*100,
                                         decimals = 2)))
    print('mean hubs percentage: {}%'.format(np.around(
											 np.mean(hubs_frac_list)*100,
                                             decimals = 2)))
    print('hubs percentage sd: {}%'.format(np.around(
										   np.std(hubs_frac_list,ddof = 1)*100,
                                           decimals = 2)))

    rand_idx_dict[null] = zrand_list

print('rand indices')

x = rand_idx_dict['sa']
y = rand_idx_dict['str']
mannwhitneyu_print(x, y, 'sa', 'str')

x = rand_idx_dict['sa']
y = rand_idx_dict['ms']
mannwhitneyu_print(x, y, 'sa', 'ms')

x = rand_idx_dict['str']
y = rand_idx_dict['ms']
mannwhitneyu_print(x, y, 'str', 'ms')

#plotting hubs fraction and z-scored rand index distributions

hub_frac_path = 'hub_frac_distribs.svg'
hub_frac_abs_path = os.path.join(distrib_path, hub_frac_path)
x = hubs_frac_dict['ms'].copy()
x.extend(hubs_frac_dict['str'])
x.extend(hubs_frac_dict['sa'])
x = np.array(x)
x = x*100
y = (['Maslov-Sneppen']*NNULLS + ['Rubinov-Sporns']*NNULLS + 
	 ['simulated annealing']*NNULLS)
ax = sns.kdeplot(x = x, hue = y, 
				 palette = ['dimgrey', '#00A1A1', '#2A7DBC'], 
				 fill = True)
ax.axvline(og_hubs_frac*100, c='orange')
ax.set_xlabel('hubs (%)')
ax.set_box_aspect(1)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), 
				ncol=3, title=None, frameon=False)
save_plot(ax, hub_frac_abs_path)

rand_idx_path = 'rand_idx_distribs.svg'
rand_idx_abs_path = os.path.join(distrib_path, rand_idx_path)
x = rand_idx_dict['ms'].copy()
x.extend(rand_idx_dict['str'])
x.extend(rand_idx_dict['sa'])
x = np.array(x)
x = x*100
ax = sns.kdeplot(x = x, hue = y, 
				 palette = ['dimgrey', '#00A1A1', '#2A7DBC'], 
				 fill = True)
ax.set_xlabel('z-Rand index')
ax.set_box_aspect(1)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), 
				ncol=3, title=None, frameon=False)
save_plot(ax, rand_idx_abs_path)

#MORPHOSPACES
morphospaces_path = os.path.join(direc, 'morphospaces')
make_dir(morphospaces_path)

print('morphospace statistics')
plt.rcParams.update({'font.size': 20})
for conn_key, SCmat in conns.items():
	print(conn_key)

	morpho_path = 'morphospace_{}.svg'.format(conn_key)
	morpho_abs_path = os.path.join(morphospaces_path, morpho_path)

	null_morpho_path = 'null_morphospace_{}.svg'.format(conn_key)
	null_morpho_abs_path = os.path.join(morphospaces_path, null_morpho_path)

	low_res_morpho_path = 'low_res_morphospace_{}.svg'.format(conn_key)
	low_res_morpho_abs_path = os.path.join(morphospaces_path, 
										   low_res_morpho_path)

	low_res_null_morpho_path='low_res_null_morphospace_{}.svg'.format(conn_key)
	low_res_null_morpho_abs_path = os.path.join(morphospaces_path, 
					     						low_res_null_morpho_path)

	data = prepro_data[conn_key]

	#calculating morphospace global network statistics
	#for the empirical network
	og_cpl = cpl_func(SCmat)
	#weight conversion between 0 and 1 to compute clustering coefficient
	conv_B = bct.weight_conversion(SCmat, 'normalize')
	og_clustering = bct.clustering_coef_wu(conv_B)
	og_mean_clustering = np.mean(og_clustering)

	data['cpl'].insert(0, og_cpl)
	data['mean_clustering'].insert(0, og_mean_clustering)
	data['colours'].insert(0, 'empirical')

	#full resolution - all nulls
	plot_morphospace(data['cpl'], data['mean_clustering'], data['colours'],
				  	 ['#6725A1', "dimgrey", "#00A1A1", "#2A7DBC"],
					 morpho_abs_path, linewidth = 0.25)

	ax = sns.jointplot(x = data['cpl'][1:], y = data['mean_clustering'][1:],
	                   hue = data['colours'][1:],
	                   palette = ["dimgrey", "#00A1A1", "#2A7DBC"], 
					   linewidth = 0.25)
	ax.plot_joint(sns.kdeplot, hue = [0, 1, 2], 
	       		  palette=['white', 'white', 'white'], 
				  zorder=1, levels=4, linewidths = 2)
	ax.ax_joint.legend_.remove()
	ax.ax_joint.set(xlabel = 'Characteristic path length',
	                ylabel = 'Clustering')
	ax.savefig(null_morpho_abs_path, dpi = 300)
	plt.close(ax.fig)

	cpl = np.array(data['cpl'])
	mean_clustering = np.array(data['mean_clustering'])
	colours = np.array(data['colours'])

	ms_idx = np.where(colours == 'Maslov-Sneppen')[0]
	str_idx = np.where(colours == 'Rubinov-Sporns')[0]
	sa_idx = np.where(colours == 'simulated annealing')[0]

	#low resolution - 100 nulls
	low_res_cpl = np.append(cpl[0], 
			 				[cpl[ms_idx][:100], 
	 						 cpl[str_idx][:100], 
							 cpl[sa_idx][:100]])
	low_res_mean_clustering = np.append(mean_clustering[0], 
				     					[mean_clustering[ms_idx][:100], 
	       								 mean_clustering[str_idx][:100], 
										 mean_clustering[sa_idx][:100]])
	low_res_colours = np.append(colours[0], 
			     				[colours[ms_idx][:100], 
								 colours[str_idx][:100], 
								 colours[sa_idx][:100]])
	plot_morphospace(low_res_cpl, low_res_mean_clustering, low_res_colours,
				  	 ["#6725A1", "dimgrey", "#00A1A1", "#2A7DBC"], 
					 low_res_morpho_abs_path)

	plot_morphospace(low_res_cpl[1:], low_res_mean_clustering[1:],
					 low_res_colours[1:], ["dimgrey", "#00A1A1", "#2A7DBC"],
					 low_res_null_morpho_abs_path)

	#subsamples
	np.random.seed(0)

	subsampling_df = {'feature': [], 'null': [], 'sample size': [], 
		   			  'mean_diff': [], 'var_diff': []}
	sample_sizes = [100, 500, 1000, 5000]
	for feature in ['cpl', 'clustering']:
		data = cpl if feature == 'cpl' else mean_clustering
		for null in ['ms', 'str', 'sa']:
			if null == 'ms':
				null_data = data[ms_idx]
			elif null == 'str':
				null_data = data[str_idx]
			else:
				null_data = data[sa_idx]
				
			#mean and variance over the whole null population
			data_mean = np.mean(null_data)
			data_var = np.var(null_data, ddof = 1)

			for n in sample_sizes:
				for i in range(1000):

					sample = np.random.choice(null_data, n, replace = False)

					#mean and variance over the subsample of size n
					sample_mean = np.mean(sample)
					sample_var = np.var(sample, ddof = 1)

					norm_mean_diff = (sample_mean - data_mean)/data_mean*100
					norm_var_diff = (sample_var - data_var)/data_var*100

					subsampling_df['feature'].append(feature)
					subsampling_df['null'].append(null)
					subsampling_df['sample size'].append(n)
					subsampling_df['mean_diff'].append(norm_mean_diff)
					subsampling_df['var_diff'].append(norm_var_diff)
	
	subsampling_df = pd.DataFrame(subsampling_df)
	for feature in ['cpl', 'clustering']:
		for stat in ['mean_diff', 'var_diff']:

			subsampling_path = 'subsampling_{}_{}_{}.svg'.format(feature, stat,
																 conn_key)
			subsampling_abs_path = os.path.join(morphospaces_path, 
									   			subsampling_path)

			data = subsampling_df[subsampling_df['feature'] == feature]
			ax = sns.lineplot(data = data, 
		     				  x = 'sample size', y = stat, hue = 'null',
						      palette = ["dimgrey", "#00A1A1", "#2A7DBC"],
						      seed = 0, legend = False)
			ax.set_xlabel('subsample size')
			if stat == 'mean_diff':
				ax.set_ylabel('relative mean difference (%)')
			else:
				ax.set_ylabel('relative variance difference (%)')
			ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), 
			 		  frameon = False)
			ax.set_box_aspect(1)
			save_plot(ax, subsampling_abs_path)

	print('CPL')

	x = cpl[sa_idx]
	y = cpl[str_idx]
	mannwhitneyu_print(x, y, 'sa', 'str')
    
	x = cpl[ms_idx]
	y = cpl[sa_idx]
	mannwhitneyu_print(x, y, 'ms', 'sa')

	x = cpl[str_idx]
	y = cpl[ms_idx]
	mannwhitneyu_print(x, y, 'str', 'ms')

	#CLUSTERING
	print('clustering')

	x = mean_clustering[sa_idx]
	y = mean_clustering[str_idx]
	mannwhitneyu_print(x, y, 'sa', 'str')

	x = mean_clustering[sa_idx]
	y = mean_clustering[ms_idx]
	mannwhitneyu_print(x, y, 'sa', 'ms')

	x = mean_clustering[str_idx]
	y = mean_clustering[ms_idx]
	mannwhitneyu_print(x, y, 'str', 'ms')

#COST-PERFORMANCE TRADEOFF
cost_path = os.path.join(direc, 'cost_perform')
make_dir(cost_path)

#tradeoff
time_path = os.path.join(cost_path, 'niter_tradeoff.svg')

niter_label = L125_niter_prepro_data['niter']
mse = L125_niter_prepro_data['mse']
time = L125_niter_prepro_data['time']

slope, intercept, r, p, _ = linregress(np.log(niter_label), mse)
print('number of iterations x MSE linear regression p-value: {}'.format(p))
line_kws = {'label':"y = {}log(x) + {}, R2 = {}".format(round(slope, 9),
                                                        round(intercept, 8),
                                                        round(r**2, 2))}
ax = sns.regplot(x = niter_label, y = mse,
                 ci = 68, seed = 0, color = "#78B7C5",
                 line_kws = line_kws, logx = True)
ax.yaxis.label.set_color("#78B7C5")
ax.tick_params(axis='y', colors="#78B7C5")
ax.set(xlabel = 'number of iterations', ylabel = 'MSE')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon = False)

ax2 = ax.twinx()
slope, intercept, r, p, _ = linregress(niter_label, time)
print('number of iterations x time linear regression p-value: {}'.format(p))
line_kws = {'label':"y = {}x + {}, R2 = {}".format(round(slope, 5),
                                                   round(intercept, 2),
                                                   round(r**2, 2))}
sns.regplot(x = niter_label, y = time,
            ci = 68, seed = 0, color = '#E1AF00',
            line_kws = line_kws, ax = ax2)
ax2.yaxis.label.set_color("#E1AF00")
ax2.tick_params(axis='y', colors="#E1AF00")
ax2.set(ylabel = 'time (s)')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), frameon = False)

align_yaxis(ax, 0, ax2, 0)

save_plot(ax, time_path)

#regplots
strengths_arr = list(zip(*Parallel(n_jobs = 6)(delayed(niter_null_strengths)(L125, seed) for seed in range(100))))
strengths_sa_1000_iter, strengths_sa_100000_iter = strengths_arr

plt.rcParams.update({'font.size': 30})
for niter, rewired_strengths in [('1000', strengths_sa_1000_iter), 
								 ('100000', strengths_sa_100000_iter)]:

	regplot_path = '{}_niter_regplot.svg'.format(niter)
	regplot_abs_path = os.path.join(cost_path, regplot_path)

	og_strengths = np.sum(L125, axis = 1)
	rewired_strengths = [strength for sublist in rewired_strengths 
					  	 for strength in sublist]

	corrs, r, sd = plot_strengths_regplots(og_strengths, rewired_strengths,
										   100, "#2A7DBC", regplot_abs_path)
    
	print('niter: {}'.format(niter))
	print('mean: {}'.format(r))
	print('sd: {}'.format(sd))

#scaling
plt.rcParams.update({'font.size': 20})
plt.rcParams['legend.fontsize'] = 15

scaling_path = 'scaling.svg'
scaling_abs_path = os.path.join(cost_path, scaling_path)

fig, ax = plt.subplots()
for null in ['ms', 'str', 'sa']:

	time_arr = []
	density_arr = []
	for filename in os.listdir('../../data/preprocessed_data/'
							   'scaling_analysis'):
		if "consensusSC_125_wei_" in filename:
			density = filename.split('_')[-2]
			conn_dict = pickle_load('../../data/preprocessed_data/'
						   			'scaling_analysis/{}'
									'_preprocessed_data_dict'.format(density))
			time_arr.extend(conn_dict['time'][null])
			#calculating density as a percentage
			density_arr.extend([(int(density)/
								(len(L125)*len(L125)-len(L125)))*100]*100)

	color = null_color(null)

	slope, intercept, r, p, _ = linregress(density_arr, time_arr)
	print('density x time ({})'
		  ' linear regression p-value: {}'.format(null, p))
	line_kws = {'label':"y = {}x + {}, R2 = {}".format(round(slope, 8),
													   round(intercept, 2),
												       round(r**2, 2))}
	sns.regplot(x = density_arr, y = time_arr, 
			 	ci = 68, seed = 0, color = color,
				line_kws = line_kws, ax = ax)
ax.set(xlabel = 'density (%)', ylabel = 'time (s)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon = False)
fig.savefig(scaling_abs_path, bbox_inches='tight', dpi = 300)
plt.close(fig)

#DIRECTED ANIMAL CONNECTOMES
dir_conn_path = os.path.join(direc, 'dir_conn')
make_dir(dir_conn_path)

for animal, SCmat in dir_conns.items():

	animal_path = os.path.join(dir_conn_path, animal + '.png')

	fig, ax = plt.subplots(figsize = (10, 10))
	sns.heatmap(SCmat, linewidths = 0.01, square = True,
				xticklabels = False, yticklabels = False,
				cmap = 'OrRd', cbar_kws = {'shrink': .5})
	save_plot(ax, animal_path)

plt.rcParams.update({'font.size': 30})
plt.rcParams['legend.fontsize'] = 20

for animal, data in dir_conn_prepro_data.items():
	SCmat = dir_conns[animal].copy()
	print(animal)

	for direction in ['in', 'out']:

		if direction == 'in':
			og_strengths = np.sum(SCmat, axis = 0)
		else:
			og_strengths = np.sum(SCmat, axis = 1)
		
		rewired_strengths = data['strengths'][direction]

		regplot_path = '{}_{}-strengths_regplot.svg'.format(animal, direction)
		regplot_abs_path = os.path.join(dir_conn_path, regplot_path)

		corrs, r, sd = plot_strengths_regplots(og_strengths, rewired_strengths,
										 	   NNULLS, "#2A7DBC", 
											   regplot_abs_path)
          
		print(direction + '-strengths')
		print('mean: {}'.format(r))
		print('sd: {}'.format(sd))

#SUPPLEMENTARY
supp_path = os.path.join(direc, 'supplementary')
make_dir(supp_path)

#max error
print('max error statistics')
for conn_key, SCmat in conns.items():

	regplot_path = 'max_error_regplot_{}.svg'.format(conn_key)
	regplot_abs_path = os.path.join(supp_path, regplot_path)

	og_strengths = np.sum(SCmat, axis = 1)

	strengths_arr = pickle_load('../../data/preprocessed_data/'
							 	'{}_maxE_strengths_sa'.format(conn_key))

	corrs, r, sd = plot_strengths_regplots(og_strengths, strengths_arr,
										   100, "#2A7DBC", regplot_abs_path)
    
	print(conn_key)
	print('mean: {}'.format(r))
	print('sd: {}'.format(sd))
    
	print('compare max error to mse')
	x = conn_corrs_dict[conn_key]['sa']
	y = corrs
	mannwhitneyu_print(x, y, 'mse', 'max error')

log_transform_corrs_dict = {}
print('log transform statistics')
for conn_key in ['L125', 'L500']:
      
	SCmat = conns[conn_key].copy()
	log_transform_corrs_dict[conn_key] = {}

	for SCmat_type in ['og', 'log']:

		#log transform
		if SCmat_type == 'log':
			values = SCmat[SCmat > 0].copy()
			values = np.log(values)
			values = scale(values, 0, 1)
			SCmat[SCmat > 0] = values

		regplot_path = '{}_{}_regplot.svg'.format(conn_key, SCmat_type)
		regplot_abs_path = os.path.join(supp_path, regplot_path)

		og_strengths = np.sum(SCmat, axis = 1)

		B_sa_list, _ = zip(*Parallel(n_jobs = 6)(delayed(strength_preserving_rand_sa)(SCmat, seed = seed) for seed in range(100)))
		rewired_strengths = []
		for B_sa in B_sa_list:
			rewired_strengths.append(np.sum(B_sa, axis = 1))
		rewired_strengths = [strength for sublist in rewired_strengths 
							 for strength in sublist]

		corrs, r, sd = plot_strengths_regplots(og_strengths, 
											   rewired_strengths,
											   100, "#2A7DBC", 
											   regplot_abs_path)
               
		log_transform_corrs_dict[conn_key][SCmat_type] = corrs
          
		print(conn_key)
		print(SCmat_type)
		print('mean: {}'.format(r))
		print('sd: {}'.format(sd))
          
print('log transform statistics')

print('L125')
x = log_transform_corrs_dict['L125']['log']
y = log_transform_corrs_dict['L125']['og']
mannwhitneyu_print(x, y, 'log', 'og')

print('L500')
x = log_transform_corrs_dict['L500']['log']
y = log_transform_corrs_dict['L500']['og']
mannwhitneyu_print(x, y, 'log', 'og')

#mse distributions
print('mse statistics')
plt.rcParams.update({'font.size': 15})
for conn_key in conn_keys:
	print(conn_key)

	mse_path = 'mse_distribs_{}.svg'.format(conn_key)
	mse_abs_path = os.path.join(supp_path, mse_path)

	data = prepro_data[conn_key]['mse']

	ax = sns.kdeplot(x = np.append(data['ms'], data['str']),
					 hue = (['Maslov-Sneppen']*NNULLS + 
							['Rubinov-Sporns']*NNULLS),
					 palette = ['dimgrey', '#00A1A1'],
					 fill = True, cut = 0, legend = False)
	ax.axvline(np.mean(data['sa']), c='#2A7DBC')
	ax.set_xlabel('MSE')
	ax.set_box_aspect(1)
	save_plot(ax, mse_abs_path)

	x = data['sa']
	y = data['str']
	mannwhitneyu_print(x, y, 'sa', 'str')

	x = data['sa']
	y = data['ms']
	mannwhitneyu_print(x, y, 'sa', 'ms')

	x = data['str']
	y = data['ms']
	mannwhitneyu_print(x, y, 'str', 'ms')
