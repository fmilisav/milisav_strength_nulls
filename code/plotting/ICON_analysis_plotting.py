import os

import numpy as np
from pingouin import compute_effsize
from scipy.stats import goodness_of_fit, shapiro, skew, \
                        skew, uniform

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import to_hex

from utils import build_morpho_dict, mannwhitneyu_print, make_dir, \
                  null_color, pickle_load, \
                  plot_morphospace, plot_strengths_regplots, save_plot

NNULLS = 100

direc = os.path.abspath('../../figures')
net_path = os.path.join(direc, 'weightedNets_test')
make_dir(net_path)

#plotting style parameters
sns.set_style("ticks")
sns.set(context=None, style=None, palette=None, font_scale=5, color_codes=None)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 20})
plt.rcParams['legend.fontsize'] = 15

#NETWORKS

net_list = list(range(38))
net_list.remove(11)
net_list = np.array(net_list) + 1

features = ['cpl', 'mean_clustering', 'colours']
nulls = ['original', 'ms', 'str', 'sa']

net_data_path = '../../data/preprocessed_data/weightedNets/net'
print('NETWORKS' + '\n--------')

#morphospaces

net_morphospace = build_morpho_dict(net_data_path, net_list)

net_morpho_path = os.path.join(net_path, 'net_morphospace.svg')
og_idx = np.where(net_morphospace['colours'] == 'original')[0]
x = net_morphospace['cpl'][og_idx]
y = net_morphospace['mean_clustering'][og_idx]
hue = net_morphospace['net'][og_idx]
palette = cm.get_cmap('jet', len(net_list))
plot_morphospace(x, y, hue, palette, net_morpho_path)

#strength sequence preservation

print('strength sequence preservation statistics')
plt.rcParams.update({'font.size': 30})
regplots_path = os.path.join(net_path, 'regplots')
make_dir(regplots_path)

mean_corrs = []
cumul_corrs = {'ms': [], 'str': [], 'sa': []}
for net in net_list:
    print('network {}'.format(net))
    file = '{}_100_nulls_preprocessed_data_dict'.format(net)
    data = pickle_load(net_data_path + file)

    net_corrs = {}
    for null in ['ms', 'str', 'sa']:

        regplot_path = 'regplot_{}_{}.svg'.format(null, net)
        regplot_abs_path = os.path.join(regplots_path, regplot_path)

        color = null_color(null)

        og_strengths = data['strengths']['original']
        rewired_strengths = data['strengths'][null]

        corrs, r, sd = plot_strengths_regplots(og_strengths, rewired_strengths,
                                               NNULLS, color, regplot_abs_path)

        cumul_corrs[null].extend(corrs)
        net_corrs[null] = np.array(corrs)

        if null == 'sa':
            mean_corrs.append(r)

    x = net_corrs['sa']
    y = net_corrs['str']
    mannwhitneyu_print(x, y, 'sa', 'str')

    x = net_corrs['sa']
    y = net_corrs['ms']
    mannwhitneyu_print(x, y, 'sa', 'ms')

    x = net_corrs['str']
    y = net_corrs['ms']
    mannwhitneyu_print(x, y, 'str', 'ms')

mean_corrs = np.array(mean_corrs)

print('cumulative strength sequence preservation statistics')
plt.rcParams.update({'font.size': 15})
plt.rcParams['legend.fontsize'] = 20
corr_path = os.path.join(net_path, 'net_strength_corr_cumul_distribs.svg')

cumul_distrib_len = NNULLS*len(net_list)
ax = sns.kdeplot(x = np.append(cumul_corrs['ms'],
                               [cumul_corrs['str'],
                                cumul_corrs['sa']]),
                    hue = (['ms']*cumul_distrib_len +
                           ['str']*cumul_distrib_len +
                           ['sa']*cumul_distrib_len),
                    palette = ['dimgrey', '#00A1A1', '#2A7DBC'],
                    fill = True, cut = 0)
ax.set_xlabel("Spearman's rho")
ax.set_box_aspect(1)
save_plot(ax, corr_path)

x = cumul_corrs['sa']
y = cumul_corrs['str']
mannwhitneyu_print(x, y, 'sa', 'str')

x = cumul_corrs['sa']
y = cumul_corrs['ms']
mannwhitneyu_print(x, y, 'sa', 'ms')

x = cumul_corrs['str']
y = cumul_corrs['ms']
mannwhitneyu_print(x, y, 'str', 'ms')

#performance factors

#building outliers colormap
cmap = cm.get_cmap('autumn_r')
colors = cmap(np.linspace(0, 1, 4))
colors = [to_hex(color) for color in colors]
colors.insert(0, 'dimgrey')

hue = ['20-100']*len(net_list)
hue = np.array(hue)
percent20_idx = np.where(mean_corrs < np.percentile(mean_corrs, 20))
hue[percent20_idx] = '15-20'
percent15_idx = np.where(mean_corrs < np.percentile(mean_corrs, 15))
hue[percent15_idx] = '10-15'
percent10_idx = np.where(mean_corrs < np.percentile(mean_corrs, 10))
hue[percent10_idx] = '5-10'
percent5_idx = np.where(mean_corrs < np.percentile(mean_corrs, 5))
hue[percent5_idx] = '0-5'
hue_order = ['20-100', '15-20', '10-15', '5-10', '0-5']

plt.rcParams.update({'font.size': 20})
plt.rcParams['legend.fontsize'] = 15

#original network stats
sw_stats = []
weight_skew_stats = []
densities = []
degree_skew_stats = []
uniform_stats = []
for net in net_list:

    print('network {}'.format(net))
    mat = np.load('../../data/original_data/'
                  'weightedNets/weightedNet{}.npy'.format(net))

    #edge stats
    nonzeros = mat.flatten()
    nonzeros = nonzeros[nonzeros != 0]

    sw = shapiro(nonzeros)[0]
    sw_stats.append(sw)

    weight_skew_stat = skew(nonzeros, bias = False)
    weight_skew_stats.append(weight_skew_stat)

    #positive weight in upper triangle
    nedges = len(np.where(np.triu(mat, k = 1) > 0)[0])
    density = nedges/(len(mat)*(len(mat)-1)/2)*100

    densities.append(density)


    #degree stats
    bin_mat = mat.copy()
    bin_mat[bin_mat != 0] = 1
    degrees = np.sum(bin_mat, axis = 0)

    degree_skew_stat = skew(degrees, bias = False)
    degree_skew_stats.append(degree_skew_stat)

    gof = goodness_of_fit(uniform, degrees, statistic='ks', random_state=0)[1]
    uniform_stats.append(gof)

sw_stats = np.array(sw_stats)
weight_skew_stats = np.array(weight_skew_stats)
densities = np.array(densities)
degree_skew_stats = np.array(degree_skew_stats)
uniform_stats = np.array(uniform_stats)

#SW x mean corrs
sw_x_perform_path = os.path.join(net_path, 'weight_sw_x_mean_corrs.svg')
fig, ax = plt.subplots()
sns.scatterplot(x = sw_stats, y = mean_corrs,
                hue = hue, hue_order = hue_order,
                palette = colors, ax = ax)
ax.set(xlabel = 'Shapiro-Wilk statistic', ylabel = 'mean Spearman\'s rho')
ax.set_box_aspect(1)
ax.legend(title='percentile', loc='upper center', bbox_to_anchor=(0.5, -0.5),
          frameon=False)
fig.savefig(sw_x_perform_path, bbox_inches='tight', dpi = 300)
plt.close(fig)

#skew x mean corrs
skew_x_perform_path = os.path.join(net_path, 'weight_skew_x_mean_corrs.svg')
fig, ax = plt.subplots()
sns.scatterplot(x = weight_skew_stats, y = mean_corrs,
                hue = hue, hue_order = hue_order,
                palette = colors, ax = ax)
ax.set(xlabel = 'Skewness', ylabel = 'mean Spearman\'s rho')
ax.set_box_aspect(1)
ax.legend(title='percentile', loc='upper center', bbox_to_anchor=(0.5, -0.5),
          frameon=False)
fig.savefig(skew_x_perform_path, bbox_inches='tight', dpi = 300)
plt.close(fig)

#density x mean corrs
density_x_perform_path = os.path.join(net_path, 'density_x_mean_corrs.svg')
fig, ax = plt.subplots()
sns.scatterplot(x = densities, y = mean_corrs,
                hue = hue, hue_order = hue_order,
                palette = colors, ax = ax)
ax.set(xlabel = 'Density', ylabel = 'mean Spearman\'s rho')
ax.set_box_aspect(1)
ax.legend(title='percentile', loc='upper center', bbox_to_anchor=(0.5, -0.5),
          frameon=False)
fig.savefig(density_x_perform_path, bbox_inches='tight', dpi = 300)
plt.close(fig)

#uniform x mean corrs
uniform_x_perform_path = os.path.join(net_path,
                                      'degree_uniform_x_mean_corrs.svg')
fig, ax = plt.subplots()
sns.scatterplot(x = uniform_stats, y = mean_corrs,
                hue = hue, hue_order = hue_order,
                palette = colors, ax = ax)
ax.set(xlabel = 'Kolmogorov-Smirnov statistic',
       ylabel = 'mean Spearman\'s rho')
ax.set_box_aspect(1)
ax.legend(title='percentile', loc='upper center', bbox_to_anchor=(0.5, -0.5),
          frameon=False)
fig.savefig(uniform_x_perform_path, bbox_inches='tight', dpi = 300)
plt.close(fig)

#skew x mean corrs
skew_x_perform_path = os.path.join(net_path, 'degree_skew_x_mean_corrs.svg')
fig, ax = plt.subplots()
sns.scatterplot(x = degree_skew_stats, y = mean_corrs,
                hue = hue, hue_order = hue_order,
                palette = colors, ax = ax)
ax.set(xlabel = 'Skewness', ylabel = 'mean Spearman\'s rho')
ax.set_box_aspect(1)
ax.legend(title='percentile', loc='upper center', bbox_to_anchor=(0.5, -0.5),
          frameon=False)
fig.savefig(skew_x_perform_path, bbox_inches='tight', dpi = 300)
plt.close(fig)

#null features

net_list = list(range(38))
net_list.remove(5)
net_list.remove(11)
net_list.remove(34)
net_list = np.array(net_list) + 1
null_features = ['cpl', 'mean_clustering', 'assortativity', 'modularity']
null_feature_heatmap = np.zeros((len(null_features), len(net_list)))

net_id = 0
for net in net_list:

    print('network {}'.format(net))
    file = '{}_100_nulls_preprocessed_data_dict'.format(net)
    supp_file = '{}_100_nulls_preprocessed_data_dict_supp'.format(net)
    data = pickle_load(net_data_path + file)
    supp_data = pickle_load(net_data_path + supp_file)

    mat = np.load('../../data/original_data/'
                  'weightedNets/weightedNet{}.npy'.format(net))


    cpl = np.array(data['cpl'])
    mean_clustering = np.array(data['mean_clustering'])
    colours = np.array(data['colours'])

    ms_idx = np.where(colours == 'Maslov-Sneppen')[0]
    sa_idx = np.where(colours == 'simulated annealing')[0]

    ms_cpl = cpl[ms_idx]
    sa_cpl = cpl[sa_idx]

    cpl_eff = compute_effsize(ms_cpl, sa_cpl, eftype = 'CLES')
    null_feature_heatmap[0, net_id] = cpl_eff

    #plot example null feature distributions
    if net in [1, 13, 22]:

        df = {'CPL': np.append(ms_cpl, sa_cpl),
              'null': ['ms']*NNULLS + ['sa']*NNULLS}
        ax = sns.kdeplot(data = df, x = 'CPL', hue = 'null',
                         fill = True, palette = ['dimgrey', null_color('sa')])
        ax.set_box_aspect(1)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(os.path.join(net_path, 'null_cpl_distribs_{}.svg'.format(net)),
                    bbox_inches='tight', dpi = 300)
        plt.close(fig)

    ms_clustering = mean_clustering[ms_idx]
    sa_clustering = mean_clustering[sa_idx]

    clustering_eff = compute_effsize(ms_clustering, sa_clustering,
                                     eftype = 'CLES')
    null_feature_heatmap[1, net_id] = clustering_eff


    assortativity = np.array(supp_data['assortativity'])
    modularity = np.array(supp_data['modularity'])

    ms_assortativity = assortativity[:NNULLS]
    sa_assortativity = assortativity[NNULLS:]

    assortativity_eff = compute_effsize(ms_assortativity, sa_assortativity,
                                        eftype = 'CLES')
    null_feature_heatmap[2, net_id] = assortativity_eff

    ms_modularity = modularity[:NNULLS]
    sa_modularity = modularity[NNULLS:]

    modularity_eff = compute_effsize(ms_modularity, sa_modularity,
                                     eftype = 'CLES')
    null_feature_heatmap[3, net_id] = modularity_eff

    net_id += 1

#plot the effect size heatmap
heatmap_path = os.path.join(net_path, 'null_feature_heatmap.svg')
fig, ax = plt.subplots(figsize = (16, 8))
sns.heatmap(null_feature_heatmap,
            xticklabels = net_list, yticklabels = null_features,
            cmap = 'RdBu_r', center = 0.5, square = True,
            cbar_kws = {'shrink': 0.25, 'label': 'CLES'})
fig.savefig(heatmap_path, bbox_inches='tight', dpi = 300)
plt.close(fig)
