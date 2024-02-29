import os
import pickle

import bct
import numpy as np
from time import process_time

from strength_preserving_rand_rs import strength_preserving_rand_rs
from strength_preserving_rand_sa import strength_preserving_rand_sa
from strength_preserving_rand_sa_energy_thresh import strength_preserving_rand_sa_energy_thresh
from strength_preserving_rand_sa_dir import strength_preserving_rand_sa_dir
from strength_preserving_rand_sa_flexE import strength_preserving_rand_sa_flexE
from rich_feeder_peripheral import rich_feeder_peripheral

def make_dir(path):
    try: os.mkdir(path)
    except OSError as error:
        print(error)
        
def pickle_dump(file, var):
    with open('../../data/preprocessed_data/' + file +
              '.pickle', 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)

#function to calculate characteristic path length
#with a negative log transform from weight to length
def cpl_func(A):

    A_len = -np.log(A)
    #shortest path length matrix
    A_dist = bct.distance_wei(A_len)[0]
    np.fill_diagonal(A_dist, np.nan)
    cpl = np.nanmean(A_dist)

    return cpl

#function to calculate characteristic path length
#with an inverse transform from weight to length
def ICON_cpl_func(A):

    A_len = 1/A
    #shortest path length matrix
    A_dist = bct.distance_wei(A_len)[0]
    np.fill_diagonal(A_dist, np.nan)
    cpl = np.nanmean(A_dist)

    return cpl

#function to calculate null statistics
def null_stats(SCmat, SCmat_strengths, conn_key, seed, null_type, analysis, 
               R = None, denom = None):

    #generating a randomized null network and timing it
    if null_type == 'ms':
        t1 = process_time()
        B, _ = bct.randmio_und_connected(SCmat, 10, seed = seed)
        t2 = process_time()
    elif null_type == 'str':
        if R is None:
            R = bct.randmio_und_connected(SCmat, 10, seed = seed)[0]
        t1 = process_time()
        B = strength_preserving_rand_rs(SCmat, R = R, seed = seed)
        t2 = process_time()
    elif null_type == 'sa':
        if R is None:
            R = bct.randmio_und_connected(SCmat, 10, seed = seed)[0]
        if analysis == 'participants':
            t1 = process_time()
            B, mse = strength_preserving_rand_sa_energy_thresh(SCmat, R = R, 
                                                               seed = seed)
            t2 = process_time()
        else:
            t1 = process_time()
            B, mse = strength_preserving_rand_sa(SCmat, R = R, seed = seed)
            t2 = process_time()

    time = t2 - t1
    strengths = np.sum(B, axis = 1)

    if null_type != 'sa':
        mse = np.mean((SCmat_strengths - strengths)**2)
    if analysis == 'ICON':
        cpl = ICON_cpl_func(B)
    else:
        cpl = cpl_func(B)
    #weight conversion between 0 and 1 to compute clustering coefficient
    conv_B = bct.weight_conversion(B, 'normalize')
    clustering = bct.clustering_coef_wu(conv_B)
    mean_clustering = np.mean(clustering)

    conn_key_check = (conn_key == 'Lausanne125' or
                      conn_key == 'Lausanne500' or
                      conn_key == 'HCP_Schaefer400' or
                      conn_key == 'HCP_Schaefer800')
    
    #binarizing the null network for rich club analysis
    B_bin = B.copy()
    B_bin[B_bin > 0] = 1

    if conn_key_check:
        rfp, pvals, rc_n = rich_feeder_peripheral(B, B_bin, stat = 'sum')
        #computing the denominator for the null phi
        if denom is None:
            mask = np.triu(np.ones(len(SCmat)), 1) > 0
            sort_weights = np.sort(SCmat[mask])
            denom = np.zeros(rc_n.shape)
            for degthresh in range(rc_n.shape[1]):
                idx = int(-rc_n[:, degthresh])
                denom[:, degthresh] = np.sum(sort_weights[idx:])
        null_phi = rfp[0]/denom
    else:
        rfp = None
        pvals = None
        null_phi = [None]
        rc_n = None

    return [B, time, mse, strengths, cpl, mean_clustering, 
            rfp, pvals, null_phi, rc_n, denom]

#function to calculate null statistics
def stats(SCmat, SCmat_strengths, conn_key, seed, analysis = 'main'):

    print('seed: {}'.format(seed))

    B_ms, time_ms, mse_ms, strengths_ms, cpl_ms, mean_clustering_ms, \
    rfp_ms, pvals_ms, null_phi_ms, \
    rc_n, denom = null_stats(SCmat, SCmat_strengths, conn_key, seed, 
                             'ms', analysis)
    B_str, time_str, mse_str, strengths_str, cpl_str, mean_clustering_str, \
    rfp_str, pvals_str, null_phi_str, \
    _, _ = null_stats(SCmat, SCmat_strengths, conn_key, seed, 
                      'str', analysis, R = B_ms, denom = denom)
    B_sa, time_sa, mse_sa, strengths_sa, cpl_sa, mean_clustering_sa, \
    rfp_sa, pvals_sa, null_phi_sa, \
    _, _ = null_stats(SCmat, SCmat_strengths, conn_key, seed, 
                      'sa', analysis, R = B_ms, denom = denom)

    return [B_ms, time_ms, mse_ms, strengths_ms, 
            cpl_ms, mean_clustering_ms, 
            rfp_ms, pvals_ms, null_phi_ms, rc_n,
            B_str, time_str, mse_str, strengths_str, 
            cpl_str, mean_clustering_str, 
            rfp_str, pvals_str, null_phi_str,
            B_sa, time_sa, mse_sa, strengths_sa, 
            cpl_sa, mean_clustering_sa, 
            rfp_sa, pvals_sa, null_phi_sa]

#function to calculate null statistics using
#simulated annealing in directed networks
def dir_stats(SCmat, seed):

    print('seed: {}'.format(seed))

    t1 = process_time()
    B_sa, mse_sa = strength_preserving_rand_sa_dir(SCmat, 
                                                   energy_type = 'mse', 
                                                   seed = seed)
    t2 = process_time()

    time = t2 - t1
    strengths_in = np.sum(B_sa, axis = 0)
    strengths_out = np.sum(B_sa, axis = 1)

    return B_sa, mse_sa, time, strengths_in, strengths_out

#function to calculate null statistics using
#simulated annealing
def sa_stats(SCmat, seed, niter = 10000):

    t1 = process_time()
    B_sa, mse = strength_preserving_rand_sa(SCmat, niter = niter, seed = seed)
    t2 = process_time()

    time = t2 - t1

    return time, mse

#function to calculate null statistics using 
#simulated annealing with the maximum absolute error objective function
def max_stats(SCmat, seed):

    B_sa, _ = strength_preserving_rand_sa_flexE(SCmat, 
                                                energy_type = 'max', 
                                                seed = seed)
    strengths_sa = np.sum(B_sa, axis = 1)

    return strengths_sa