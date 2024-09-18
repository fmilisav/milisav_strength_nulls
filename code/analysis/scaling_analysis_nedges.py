import bct
import numpy as np
from time import process_time

from strength_preserving_rand_rs import strength_preserving_rand_rs
from strength_preserving_rand_sa import strength_preserving_rand_sa

import os
from utils import make_dir, pickle_dump

direc = os.path.abspath('../../data/preprocessed_data/')
scaling_path = os.path.join(direc, 'scaling_analysis')
make_dir(scaling_path)

conns = {}
for filename in os.listdir('../../data/preprocessed_data/scaling_analysis/'):
    if "consensusSC_125_wei_" in filename:
        conns[filename.split('_')[-2]] =np.load('../../data/preprocessed_data/'
                                                'scaling_analysis/' + filename)

for conn_key, SCmat in conns.items():

    conn_dict = {}

    SCmat_strengths = np.sum(SCmat, axis = 1)

    time = {'ms': [], 'str': [], 'sa': []}
    mse = {'ms': [], 'str': [], 'sa': []}

    for i in range(100):

        t1 = process_time()
        B_ms, eff = bct.randmio_und_connected(SCmat, itr = 10, seed = i)
        t2 = process_time()
        time['ms'].append(t2 - t1)
        mse['ms'].append(np.mean((np.sum(B_ms, axis = 1) -
                                  SCmat_strengths)**2))

        t1 = process_time()
        B_str = strength_preserving_rand_rs(SCmat, R = B_ms, seed = i)
        t2 = process_time()
        time['str'].append(t2 - t1)
        mse['str'].append(np.mean((np.sum(B_str, axis = 1) -
                                   SCmat_strengths)**2))

        t1 = process_time()
        B_sa, mse_sa = strength_preserving_rand_sa(SCmat, R = B_ms, seed = i)
        t2 = process_time()
        time['sa'].append(t2 - t1)
        mse['sa'].append(mse_sa)

    conn_dict['time'] = time
    conn_dict['mse'] = mse

    pickle_dump('/scaling_analysis/'
                '{}_preprocessed_data_dict'.format(conn_key), conn_dict)
