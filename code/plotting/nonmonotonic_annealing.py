import numpy as np
import pandas as pd
from trajectories_utils_fast import strength_preserving_rand_sa_trajectory
from trajectories_utils_reheat import strength_preserving_rand_sa_reheat_trajectory

import seaborn as sns
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('pickle')

import os
from utils import make_dir, mannwhitneyu_print, pickle_load

direc = os.path.abspath('../../figures')

NNULLS = 100

nm_anneal_path = os.path.join(direc, 'nm_anneal')
make_dir(nm_anneal_path)

SCmat = np.load('../../data/original_data/consensusSC_125_wei.npy')

#getting the energy at the end of each stage for the monotonic annealing
energy_trajectory, min_energy_trajectory = zip(*Parallel(n_jobs = 3)(delayed(strength_preserving_rand_sa_trajectory)(SCmat, seed = seed) for seed in range(NNULLS)))

og_min_energy_trajectory = np.array(pickle_load('../../data/preprocessed_data/'
                                                'L125_energy_trajectory_sa'))

df = pd.DataFrame({'steps': np.tile(np.arange(100), 2),
                   'energy': np.concatenate([og_min_energy_trajectory[0],
                                             energy_trajectory[0]]),
                   'hue': ['minimum']*100 + ['current']*100})

#plotting the current and minimum energy trajectories
og_trajectory_path = 'og_energy_trajectories_L125.svg'
og_trajectory_abs_path = os.path.join(nm_anneal_path, og_trajectory_path)
ax = sns.lineplot(data = df, x = 'steps', y = 'energy', hue = 'hue')
ax.set(xlabel = 'stages', ylabel = 'energy (MSE)')
ax.ticklabel_format(style = 'sci', axis = 'y', scilimits = (0,0))
fig = ax.get_figure()
fig.savefig(og_trajectory_abs_path, dpi = 300)
plt.close(fig)

#getting the non-monotonic annealing (reheating) trajectories
#for energy, minimum energy, and temperature
sa_stats_arr = zip(*Parallel(n_jobs = 6)(delayed(strength_preserving_rand_sa_reheat_trajectory)(SCmat, seed = seed, nstage = 1000, niter = 1000, frac = 0.9) for seed in range(NNULLS)))
energy_trajectory, min_energy_trajectory, temperature_trajectory = sa_stats_arr

df = pd.DataFrame({'steps': np.tile(np.arange(1000), 2),
                   'energy': np.append(min_energy_trajectory[0],
                                       energy_trajectory[0]),
                   'hue': ['minimum']*1000 + ['current']*1000})

#plotting the current and minimum energy trajectories
reheat_trajectory_path = 'reheat_energy_trajectories_L125.svg'
reheat_trajectory_abs_path = os.path.join(nm_anneal_path,
                                          reheat_trajectory_path)
ax = sns.lineplot(data = df, x = 'steps', y = 'energy', hue = 'hue')
ax.set(xlabel = 'stages', ylabel = 'energy (MSE)')
ax.ticklabel_format(style = 'sci', axis = 'y', scilimits = (0,0))
fig = ax.get_figure()
fig.savefig(reheat_trajectory_abs_path, dpi = 300)
plt.close(fig)

#plotting the temperature trajectory
temperature_trajectory_path = 'temperature_trajectory_L125.svg'
temperature_trajectory_abs_path = os.path.join(nm_anneal_path,
                                               temperature_trajectory_path)
ax = sns.lineplot(x = np.arange(1000), y = temperature_trajectory[0])
ax.set(xlabel = 'stages', ylabel = 'temperature')
fig = ax.get_figure()
fig.savefig(temperature_trajectory_abs_path, dpi = 300)
plt.close(fig)

og_energymins = []
energymins = []
for i in range(100):
    og_energymins.append(og_min_energy_trajectory[i][-1])
    energymins.append(min_energy_trajectory[i][-1])

#comparing the minimum energies of the original and reheat annealing
mannwhitneyu_print(og_energymins, energymins, 'original', 'reheat')

energymins_dict = {'energy': np.append(og_energymins, energymins),
                   'hue': ['original']*100 + ['reheat']*100}

energymins_path = 'energymins_L125.svg'
energymins_abs_path = os.path.join(nm_anneal_path, energymins_path)
ax = sns.kdeplot(data = pd.DataFrame(energymins_dict), x = 'energy',
                 hue = 'hue', palette = ['#2A7DBC', '#EF3C29'],
                 fill = True, cut = 0)
ax.set(xlabel = 'energy (MSE)')
ax.set_box_aspect(1)
fig = ax.get_figure()
fig.savefig(energymins_abs_path, dpi = 300)

#getting the minimum energies of the slow monotonic annealing
#(same number of iterations as the reheat annealing)
slow_mono_energy_trajectory, slow_mono_min_energy_trajectory = zip(*Parallel(n_jobs = 6)(delayed(strength_preserving_rand_sa_trajectory)(SCmat, seed = seed, nstage = 1000, niter = 1000, frac = 0.9) for seed in range(NNULLS)))

slow_mono_min_energymins = []
for i in range(100):
    slow_mono_min_energymins.append(slow_mono_min_energy_trajectory[i][-1])

mannwhitneyu_print(slow_mono_min_energymins, energymins,
                   'slow monotonic', 'reheat')