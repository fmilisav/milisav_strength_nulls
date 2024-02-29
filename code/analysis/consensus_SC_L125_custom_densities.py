import numpy as np
from scipy.spatial.distance import *

import bct
from struct_consensus import struct_consensus

import os
from utils import make_dir

direc = os.path.abspath('../../data/preprocessed_data/')
scaling_path = os.path.join(direc, 'scaling_analysis')
make_dir(scaling_path)

#loading Euclidean distance matrix
euc_dist = np.load('../../data/original_data/'
                   'euc_distL125.npy')

#loading cortical indices
cor_path = ('../../data/original_data/'
            'Lausanne/cortical/cortical125.txt')
cortical = np.loadtxt(os.path.abspath(cor_path))
cor_idx = [i for i, val in enumerate(cortical) if val == 1]

#loading cortical structural connectivity matrices
struct_den = np.load('../../data/original_data/'
                     'Lausanne/struct/struct_den_scale125.npy')
cor_struct_den = struct_den[cor_idx][:, cor_idx]
#computing mean weighted SC across subjects
mean = np.mean(cor_struct_den, axis = 2)

#building hemispheric labels for struct_consensus
first_right_sub_id = np.argwhere(cortical == 0)[0, 0]
len_left = len(cor_idx) - first_right_sub_id
hemiid = np.array(first_right_sub_id*[0] + len_left*[1])
hemiid = hemiid[:, np.newaxis]

inter_n = 508 #average number of inter-hemispheric connections
inter_n_arr = [508]
intra_n = 2132 #average number of intra-hemispheric connections
intra_n_arr = [2132]

#upscale/downscale inter/intra-hemispheric connections
while inter_n + intra_n < 20000: 
    inter_n = inter_n*1.10
    intra_n = intra_n*1.10
    inter_n_arr.append(inter_n)
    intra_n_arr.append(intra_n)
inter_n = 508
intra_n = 2132
while inter_n + intra_n > 1900:
    inter_n = inter_n/1.10
    intra_n = intra_n/1.10
    inter_n_arr.insert(0, inter_n)
    intra_n_arr.insert(0, intra_n)

#generate consensus SCs for different densities
for i in range(len(inter_n_arr)):

    inter_nedges = inter_n_arr[i]
    intra_nedges = intra_n_arr[i]

    consensusSC_bin = struct_consensus(cor_struct_den, euc_dist, hemiid,
                                       conn_num_inter = inter_nedges,
                                       conn_num_intra = intra_nedges)
    eff_nedges = np.sum(consensusSC_bin > 0) #effective number of edges

    #check if consensus SC is disconnected
    if bct.number_of_components(consensusSC_bin) > 1:
        print(str(eff_nedges) + ': disconnected')
        break

    #weight consensus SC by mean across subjects
    consensusSC_wei = consensusSC_bin * mean

    with open('../../data/preprocessed_data/scaling_analysis/'
              'consensusSC_125_wei_{}_conns.npy'.format(eff_nedges), 
              'wb') as f:
        np.save(f, consensusSC_wei)
