import os

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils import build_morpho_dict, mannwhitneyu_print, make_dir, \
                  null_color, pickle_load, \
                  plot_morphospace, plot_strengths_regplots, save_plot

NNULLS = 100

direc = os.path.abspath('../../figures')
net_path = os.path.join(direc, 'weightedNets')
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

    x = net_corrs['sa']
    y = net_corrs['str']
    mannwhitneyu_print(x, y, 'sa', 'str')

    x = net_corrs['sa']
    y = net_corrs['ms']
    mannwhitneyu_print(x, y, 'sa', 'ms')

    x = net_corrs['str']
    y = net_corrs['ms']
    mannwhitneyu_print(x, y, 'str', 'ms')

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