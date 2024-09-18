import numpy as np

from utils import signed_stats, pickle_dump

from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('pickle')

import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

NNULLS = 100

signed_conns = {}
for filename in os.listdir('../../data/original_data/many_networks/'):
    signed_conns[filename.split('.')[0]] = np.load('../../data/original_data/'
                                                   'many_networks/' + filename)

signed_conn_prepro_data_dict = {}
for key, net in signed_conns.items():

	signed_conn_prepro_data_dict = {}
	signed_conn_prepro_data_dict['mse_sa'] = []
	signed_conn_prepro_data_dict['time'] = {'ms': [], 'sa': []}
	signed_conn_prepro_data_dict['strengths'] = {'ms': {}, 'sa': {}}

	stats_arr = list(zip(*Parallel(n_jobs = 50)(delayed(signed_stats)(net, seed) for seed in range(NNULLS))))
	_, time_ms, strengths_pos_ms, strengths_neg_ms, \
	_, time_sa, mse_sa, strengths_pos_sa, strengths_neg_sa = stats_arr

	signed_conn_prepro_data_dict['mse_sa'].extend(mse_sa)

	signed_conn_prepro_data_dict['time']['ms'].extend(time_ms)
	signed_conn_prepro_data_dict['time']['sa'].extend(time_sa)

	for null_type in ['ms', 'sa']:
		for sign in ['pos', 'neg']:
			signed_conn_prepro_data_dict['strengths'][null_type][sign] = []

	signed_conn_prepro_data_dict['strengths']['ms']['pos'].extend(np.concatenate(strengths_pos_ms))
	signed_conn_prepro_data_dict['strengths']['ms']['neg'].extend(np.concatenate(strengths_neg_ms))
	signed_conn_prepro_data_dict['strengths']['sa']['pos'].extend(np.concatenate(strengths_pos_sa))
	signed_conn_prepro_data_dict['strengths']['sa']['neg'].extend(np.concatenate(strengths_neg_sa))

	pickle_dump('{}_prepro_data_dict'.format(key), signed_conn_prepro_data_dict)