import numpy as np
from scipy.spatial.distance import *

import bct
from struct_consensus import struct_consensus

import os
from utils import make_dir

direc = os.path.abspath('../../data/preprocessed_data/')
scaling_path = os.path.join(direc, 'scaling_analysis')
make_dir(scaling_path)

resolutions = ['060', '125', '250', '500']
for res in resolutions:

    cor_path = ('../../data/original_data/'
                'Lausanne/cortical/cortical{}.txt'.format(res))
    cortical = np.loadtxt(os.path.abspath(cor_path))
    cor_idx = [i for i, val in enumerate(cortical) if val == 1]

    coords = np.load('../../data/original_data/'
                     'Lausanne/coords/coords{}.npy'.format(res))
    cor_coords = coords[cor_idx]
    euc_dist = squareform(pdist(cor_coords))

    struct_den = np.load('../../data/original_data/'
                         'Lausanne/struct/struct_den_scale{}.npy'.format(res))
    cor_struct_den = struct_den[cor_idx][:, cor_idx]
    mean = np.mean(cor_struct_den, axis = 2)

    first_right_sub_id = np.argwhere(cortical == 0)[0, 0]
    len_left = len(cor_idx) - first_right_sub_id
    hemiid = np.array(first_right_sub_id*[0] + len_left*[1])
    hemiid = hemiid[:, np.newaxis]

    consensusSC_bin = struct_consensus(cor_struct_den, euc_dist, hemiid,
                                       avg_conn_num_inter = 1993,
                                       avg_conn_num_intra = 8793)
    if bct.number_of_components(consensusSC_bin) > 1:
        print(res + ': disconnected')
    consensusSC_wei = consensusSC_bin * mean

    with open(os.path.join(scaling_path,
              'consensusSC_{}_wei_10786_conns.npy'.format(res)), 'wb') as f:
        np.save(f, consensusSC_wei)