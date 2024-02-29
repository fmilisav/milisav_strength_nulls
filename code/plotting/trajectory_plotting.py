import os

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from utils import pickle_load, make_dir, save_plot

direc = os.path.abspath('../../figures')
trajectory_path = os.path.join(direc, 'trajectory')
make_dir(trajectory_path)

#plotting style parameters
sns.set_style("ticks")
sns.set(context=None, style=None, palette=None, 
        font_scale=5, color_codes=None, rc=None)
plt.rcParams['svg.fonttype'] = "none"
plt.rcParams.update({'font.size': 20})

for conn in ['L125', 'HCP400']:

    #loading preprocessed data
    strengths_trajectory = np.array(pickle_load('../../data/preprocessed_data/'
                                    '{}_strengths_trajectory_sa'.format(conn)))
    energy_trajectory = np.array(pickle_load('../../data/preprocessed_data/'
                                 '{}_energy_trajectory_sa'.format(conn)))
    cpl_trajectory = np.array(pickle_load('../../data/preprocessed_data/'
                              '{}_cpl_trajectory_sa'.format(conn)))
    clustering_trajectory=np.array(pickle_load('../../data/preprocessed_data/'
                                   '{}_clustering_trajectory_sa'.format(conn)))

    #energy morphospace
    df = {'cpl': cpl_trajectory.flatten(), 
          'clustering': clustering_trajectory.flatten(), 
          'energy': energy_trajectory.flatten()}
    df = pd.DataFrame(df)
    df['cpl_bin'] = pd.cut(df['cpl'], bins = 50)
    df['clustering_bin'] = pd.cut(df['clustering'], bins = 50)
    mid_df = df.copy()
    mid_df['cpl_bin_mid'] = [0]*len(mid_df)
    mid_df['clustering_bin_mid'] = [0]*len(mid_df)
    for i in range(len(mid_df)):
        mid_df.loc[i, 'cpl_bin_mid'] = mid_df.loc[i, 'cpl_bin'].mid
        mid_df.loc[i,'clustering_bin_mid'] = mid_df.loc[i,'clustering_bin'].mid
    grouped = mid_df.groupby(['cpl_bin_mid', 'clustering_bin_mid'], 
                             as_index=False)['energy'].mean()
    grouped = grouped.round(6)
    data = grouped.pivot('clustering_bin_mid', 'cpl_bin_mid', 'energy')

    meanE_morpho_path = os.path.join(trajectory_path, 
                                     '{}_mean_energy_morphospace'
                                     '.svg'.format(conn))
    ax = sns.heatmap(data, cmap=sns.cm.mako_r, 
                     cbar_kws = {'label': 'Mean energy (MSE)'})
    ax.invert_yaxis()
    ax.set(xlabel = 'Characteristic path length', ylabel = 'Clustering')
    ax.set_box_aspect(1)
    save_plot(ax, meanE_morpho_path)

    #trajectory
    prepro_data_key = 'Lausanne125' if conn == 'L125' else 'HCP_Schaefer400'
    prepro_data = pickle_load('../../data/preprocessed_data/'
                        '{}_preprocessed_data_dict'.format(prepro_data_key))
    colours = np.array(prepro_data['colours'])
    ms_idx = np.where(colours == 'Maslov-Sneppen')[0]
    sa_idx = np.where(colours == 'simulated annealing')[0]
    cpl = np.array(prepro_data['cpl'])
    cpl = np.append(cpl[ms_idx], cpl[sa_idx])
    mean_clustering = np.array(prepro_data['mean_clustering'])
    mean_clustering = np.append(mean_clustering[ms_idx], 
                                mean_clustering[sa_idx])
    colours = np.append(colours[ms_idx], colours[sa_idx])

    morpho_trajectory_path = os.path.join(trajectory_path, 
                                          '{}_morphospace_trajectory_1'
                                          '.svg'.format(conn))
    ax = sns.jointplot(cpl, mean_clustering, 
                       hue = colours, palette = ["dimgrey", "#2A7DBC"],
                       rasterized = True, alpha = 0.1, linewidth = 0.25)
    x = cpl_trajectory[0]
    y = clustering_trajectory[0]
    last_cpl = x[-1]
    last_clustering = y[-1]
    #https://stackoverflow.com/questions/8500700
    path = mpath.Path(np.column_stack([x, y]))
    verts = path.interpolated(steps=3).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(0, 1, len(x))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lines = mcoll.LineCollection(segments, array=z, 
                                 cmap=plt.get_cmap('RdBu_r'),
                                 norm=plt.Normalize(0.0, 1.0), 
                                 linewidth=2, alpha=1.0)
    cpl_diff = 0.01
    clustering_diff = 0.0001 if conn == 'L125' else 0.0005
    ax.ax_joint.add_patch(Rectangle((last_cpl - cpl_diff,
                                     last_clustering - clustering_diff), 
                                     cpl_diff*2, clustering_diff*2,
                                     lw = 2, edgecolor = 'k', 
                                     alpha = 0.7, facecolor = 'none'))
    ax.ax_joint.add_collection(lines)
    ax.ax_joint.set(xlabel = 'Characteristic path length', 
                    ylabel = 'Clustering')
    ax.ax_joint.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), 
                       frameon = False)
    ax.savefig(morpho_trajectory_path, dpi = 300)
    plt.close(ax.fig)
        
    morpho_trajectory_path = os.path.join(trajectory_path, 
                                          '{}_morphospace_trajectory_2'
                                          '.svg'.format(conn))
    ax = sns.scatterplot(cpl, mean_clustering, 
                         hue = colours, palette = ["dimgrey", "#2A7DBC"],
                         legend = False, rasterized = True, alpha = 0.25)
    lines = mcoll.LineCollection(segments, array=z, 
                                 cmap=plt.get_cmap('RdBu_r'), 
                                 norm=plt.Normalize(0.0, 1.0), 
                                 linewidth=2, alpha=1.0)
    ax.add_collection(lines)
    ax.set(xlabel = 'Characteristic path length', ylabel = 'Clustering')
    ax.set_xlim(last_cpl - cpl_diff, last_cpl + cpl_diff)
    ax.set_ylim(last_clustering - clustering_diff, 
                last_clustering + clustering_diff)
    cpl_diff = cpl_diff/5
    clustering_diff = clustering_diff/5
    ax.add_patch(Rectangle((last_cpl - cpl_diff, 
                            last_clustering - clustering_diff), 
                            cpl_diff*2, clustering_diff*2, 
                            lw = 2, edgecolor = 'k', 
                            alpha = 0.7, facecolor = 'none'))
    ax.set_box_aspect(1)
    save_plot(ax, morpho_trajectory_path, close = False)

    for i in range(3, 6):

        morpho_trajectory_path = os.path.join(trajectory_path, 
                                            '{}_morphospace_trajectory_{}'
                                            '.svg'.format(conn, i))
        ax.set_xlim(last_cpl - cpl_diff, last_cpl + cpl_diff)
        ax.set_ylim(last_clustering - clustering_diff, 
                    last_clustering + clustering_diff)
        cpl_diff = cpl_diff/5
        clustering_diff = clustering_diff/5
        if i < 5:
            ax.add_patch(Rectangle((last_cpl - cpl_diff, 
                                    last_clustering - clustering_diff), 
                                    cpl_diff*2, clustering_diff*2, 
                                    lw = 2, edgecolor = 'k', 
                                    alpha = 0.7, facecolor = 'none'))
        ax.set_box_aspect(1)
        close = False if i < 5 else True
        save_plot(ax, morpho_trajectory_path, close = close)

    #energy trajectory
    energy = []
    stages = []
    for step in range(energy_trajectory.shape[1]):
        energy.extend(energy_trajectory[:, step])
        stages.extend([step]*energy_trajectory.shape[0])

    energy_trajectory_path = os.path.join(trajectory_path, 
                                          '{}_energy_trajectory'
                                          '.svg'.format(conn))
    ax = sns.scatterplot(x = stages, y = energy, 
                         c = mcolors.to_rgb("#78B7C5"), 
                         rasterized = True)
    ax.set(xlabel = 'stages', ylabel = 'energy (MSE)')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    save_plot(ax, energy_trajectory_path)