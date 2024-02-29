import numpy as np

from utils import sa_stats, pickle_dump

conns = {}
for res in ['125']:
    conns['Lausanne{}'.format(res)] = np.load('../../data/original_data/'
                                              'consensusSC_{}_wei'
                                              '.npy'.format(res))

niter_arr = [1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 100000]

for conn_key, SCmat in conns.items():

  time = []
  mse = []
  niter_label = []
  for niter in niter_arr:
    print(niter)
    for seed in range(100):
        time_sa, mse_sa = sa_stats(SCmat, seed, niter = niter)
        time.append(time_sa)
        mse.append(mse_sa)
        niter_label.append(niter)

  conn_dict = {'time': time, 'mse': mse, 'niter': niter_label}
  pickle_dump('{}_niter_preprocessed_data_dict'.format(conn_key), conn_dict)
