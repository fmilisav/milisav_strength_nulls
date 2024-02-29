import os

import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils import build_morpho_dict, make_dir, mannwhitneyu_print, \
                  null_color, pickle_load, plot_morphospace, \
                  plot_strengths_regplots, save_plot

NNULLS = 100

direc = os.path.abspath('../../figures')
subj_path = os.path.join(direc, 'subj-wise_analysis')
make_dir(subj_path)

#plotting style parameters
sns.set_style("ticks")
sns.set(context=None, style=None, palette=None, font_scale=5, color_codes=None)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 20})
plt.rcParams['legend.fontsize'] = 15

#SUBJECTS

subj_list = list(range(70))
subj_list.pop(8)

subj_data_path = '../../data/preprocessed_data/subjects/subj'
print('SUBJECTS' + '\n--------')

#feature dictionary
subj_morphospace = build_morpho_dict(subj_data_path, subj_list)

#morphospaces

#all networks
subj_morpho_path = os.path.join(subj_path, 'subj_morphospace.svg')
x = subj_morphospace['cpl']
y = subj_morphospace['mean_clustering']
hue = subj_morphospace['net']
palette = cm.get_cmap('jet', len(subj_list))
plot_morphospace(x, y, hue, palette, subj_morpho_path, linewidth = 0.5,
                 plot_legend = True, legend_title = 'subjects')

#all nulls

plt.rcParams.update({'font.size': 20})
subj_morpho_path = os.path.join(subj_path, 'subj_morphospace_nulls.svg')

null_idx = np.where(subj_morphospace['colours'] != 'original')[0]
ms_idx = np.where(subj_morphospace['colours'] == 'Maslov-Sneppen')[0]
str_idx = np.where(subj_morphospace['colours'] == 'Rubinov-Sporns')[0]
sa_idx = np.where(subj_morphospace['colours'] == 'simulated annealing')[0]

x = subj_morphospace['cpl'][null_idx]
y = subj_morphospace['mean_clustering'][null_idx]
hue = subj_morphospace['colours'][null_idx]
palette = ["dimgrey", "#00A1A1", "#2A7DBC"]
plot_morphospace(x, y, hue, palette, subj_morpho_path, linewidth = 0.5,
                 plot_legend = True, legend_title = 'null')

#Maslov-Sneppen

subj_morpho_path = os.path.join(subj_path, 'subj_morphospace_ms.svg')

x = subj_morphospace['cpl'][ms_idx]
y = subj_morphospace['mean_clustering'][ms_idx]
hue = subj_morphospace['net'][ms_idx]
palette = cm.get_cmap('jet', len(subj_list))
plot_morphospace(x, y, hue, palette, subj_morpho_path, linewidth = 0.5,
                 plot_legend = True, legend_title = 'subjects')

#Rubinov-Sporns

subj_morpho_path = os.path.join(subj_path, 'subj_morphospace_str.svg')

x = subj_morphospace['cpl'][str_idx]
y = subj_morphospace['mean_clustering'][str_idx]
hue = subj_morphospace['net'][str_idx]
plot_morphospace(x, y, hue, palette, subj_morpho_path, linewidth = 0.5,
                 plot_legend = True, legend_title = 'subjects')

#simulated annealing

subj_morpho_path = os.path.join(subj_path, 'subj_morphospace_sa.svg')

x = subj_morphospace['cpl'][sa_idx]
y = subj_morphospace['mean_clustering'][sa_idx]
hue = subj_morphospace['net'][sa_idx]
plot_morphospace(x, y, hue, palette, subj_morpho_path, linewidth = 0.5,
                 plot_legend = True, legend_title = 'subjects')

#subjects relationships preservation

#coordinates of original networks in morphospace
original = subj_morphospace['colours'] == 'original'
X = list(zip(subj_morphospace['cpl'][original],
             subj_morphospace['mean_clustering'][original]))

#Euclidean distance between original networks
og_subj_dist = squareform(pdist(X))
#upper triangle
subj_dist_triu_idx = np.triu_indices(og_subj_dist.shape[0], k = 1)
og_subj_dist_triu = og_subj_dist[subj_dist_triu_idx]

#for each seed and null model, 
#calculating null Euclidean distance matrix between subjects
corr_dict = {'ms': [], 'str': [], 'sa': []}
for i in range(NNULLS):
    for null in ['ms', 'str', 'sa']:
        if null == 'ms':
            colour = 'Maslov-Sneppen'
        elif null == 'str':
            colour = 'Rubinov-Sporns'
        else: colour = 'simulated annealing'
        null_idx = subj_morphospace['colours'] == colour
        
        #coordinates of null networks in morphospace
        cpl = []
        mean_clustering = []
        for subj in subj_list:
            subj_idx = subj_morphospace['net'] == subj
            idx = null_idx & subj_idx
            cpl.append(subj_morphospace['cpl'][idx][i])
            mean_clustering.append(subj_morphospace['mean_clustering'][idx][i])
        
        X = list(zip(cpl, mean_clustering))
        #Euclidean distance between null networks
        null_subj_dist = squareform(pdist(X))
        #upper triangle
        null_subj_dist_triu = null_subj_dist[subj_dist_triu_idx]
        
        corr_dict[null].append(spearmanr(og_subj_dist_triu, 
                                         null_subj_dist_triu)[0])

print('subject relationships preservation statistics')
plt.rcParams.update({'font.size': 15})
plt.rcParams['legend.fontsize'] = 20
corr_path = os.path.join(subj_path, 'subj_corr_distribs.svg')

ax = sns.kdeplot(x = np.append(corr_dict['ms'], 
                               [corr_dict['str'], corr_dict['sa']]),
                    hue = (['Maslov-Sneppen']*NNULLS + 
                           ['Rubinov-Sporns']*NNULLS + 
                           ['simulated annealing']*NNULLS),
                    palette = ['dimgrey', '#00A1A1', '#2A7DBC'],
                    fill = True, cut = 0)
ax.set_xlabel("Spearman's rho")
ax.set_box_aspect(1)
save_plot(ax, corr_path)

x = np.array(corr_dict['sa'])
y = np.array(corr_dict['str'])
mannwhitneyu_print(x, y, 'sa', 'str')

x = corr_dict['sa']
y = corr_dict['ms']
mannwhitneyu_print(x, y, 'sa', 'ms')

x = np.array(corr_dict['str'])
y = np.array(corr_dict['ms'])
mannwhitneyu_print(x, y, 'str', 'ms')


#strength sequence preservation

print('strength sequence preservation statistics')
plt.rcParams.update({'font.size': 30})
regplots_path = os.path.join(subj_path, 'regplots')
make_dir(regplots_path)

cumul_corrs = {'ms': [], 'str': [], 'sa': []}
for subj in subj_list:
    print('subject {}'.format(subj))
    file = '{}_100_nulls_preprocessed_data_dict'.format(subj)
    data = pickle_load(subj_data_path + file)

    subj_corrs = {}
    for null in ['ms', 'str', 'sa']:

        regplot_path = 'regplot_{}_{}.svg'.format(null, subj)
        regplot_abs_path = os.path.join(regplots_path, regplot_path)

        color = null_color(null)
        
        og_strengths = data['strengths']['original']
        rewired_strengths = data['strengths'][null]

        corrs, r, sd = plot_strengths_regplots(og_strengths, rewired_strengths,
                                               NNULLS, color, regplot_abs_path)

        
        cumul_corrs[null].extend(corrs)
        subj_corrs[null] = np.array(corrs)

    x = subj_corrs['sa']
    y = subj_corrs['str']
    mannwhitneyu_print(x, y, 'sa', 'str')

    x = subj_corrs['sa']
    y = subj_corrs['ms']
    mannwhitneyu_print(x, y, 'sa', 'ms')

    x = subj_corrs['str']
    y = subj_corrs['ms']
    mannwhitneyu_print(x, y, 'str', 'ms')

print('cumulative strength sequence preservation statistics')
plt.rcParams.update({'font.size': 15})
plt.rcParams['legend.fontsize'] = 20
corr_path = os.path.join(subj_path, 'subj_strength_corr_cumul_distribs.svg')

cumul_distrib_len = NNULLS*len(subj_list)
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


#morphospaces

print('morphospace statistics')
plt.rcParams.update({'font.size': 20})
plt.rcParams['legend.fontsize'] = 15
morphospaces_path = os.path.join(subj_path, 'morphospaces')
make_dir(morphospaces_path)

for subj in subj_list:
    print('subject {}'.format(subj))
    subj_idx = subj_morphospace['net'] == subj

    morpho_path = 'morphospace_{}.svg'.format(subj)
    morpho_abs_path = os.path.join(morphospaces_path, morpho_path)

    null_morpho_path = 'null_morphospace_{}.svg'.format(subj)
    null_morpho_abs_path = os.path.join(morphospaces_path, null_morpho_path)

    og_idx = subj_morphospace['colours'] == 'original'
    ms_idx = subj_morphospace['colours'] == 'Maslov-Sneppen'
    str_idx = subj_morphospace['colours'] == 'Rubinov-Sporns'
    sa_idx = subj_morphospace['colours'] == 'simulated annealing'

    subj_og_idx = subj_idx & og_idx
    subj_ms_idx = subj_idx & ms_idx
    subj_str_idx = subj_idx & str_idx
    subj_sa_idx = subj_idx & sa_idx

    x = subj_morphospace['cpl'][subj_idx]
    y = subj_morphospace['mean_clustering'][subj_idx]
    hue = subj_morphospace['colours'][subj_idx]
    palette = ["#6725A1", "dimgrey", "#00A1A1", "#2A7DBC"]
    plot_morphospace(x, y, hue, palette, morpho_abs_path)

    x = subj_morphospace['cpl'][subj_idx][1:]
    y = subj_morphospace['mean_clustering'][subj_idx][1:]
    hue = subj_morphospace['colours'][subj_idx][1:]
    palette = ["dimgrey", "#00A1A1", "#2A7DBC"]
    plot_morphospace(x, y, hue, palette, null_morpho_abs_path)

    #CPL
    print('CPL')

    x = subj_morphospace['cpl'][subj_sa_idx]
    y = subj_morphospace['cpl'][subj_str_idx]
    mannwhitneyu_print(x, y, 'sa', 'str')

    x = subj_morphospace['cpl'][subj_sa_idx]
    y = subj_morphospace['cpl'][subj_ms_idx]
    mannwhitneyu_print(x, y, 'sa', 'ms')

    x = subj_morphospace['cpl'][subj_str_idx]
    y = subj_morphospace['cpl'][subj_ms_idx]
    mannwhitneyu_print(x, y, 'str', 'ms')

    #CLUSTERING
    print('clustering')

    x = subj_morphospace['mean_clustering'][subj_sa_idx]
    y = subj_morphospace['mean_clustering'][subj_str_idx]
    mannwhitneyu_print(x, y, 'sa', 'str')

    x = subj_morphospace['mean_clustering'][subj_sa_idx]
    y = subj_morphospace['mean_clustering'][subj_ms_idx]
    mannwhitneyu_print(x, y, 'sa', 'ms')

    x = subj_morphospace['mean_clustering'][subj_str_idx]
    y = subj_morphospace['mean_clustering'][subj_ms_idx]
    mannwhitneyu_print(x, y, 'str', 'ms')
