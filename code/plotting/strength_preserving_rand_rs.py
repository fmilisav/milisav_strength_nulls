import bct
import math
import numpy as np
from sklearn.utils import check_random_state

def strength_preserving_rand_rs(A,
                                rewiring_iter = 10, sort_freq = 1,
                                R = None, connected = None, 
                                seed = None):
    """
    Degree- and strength-preserving randomization of
    undirected, weighted adjacency matrix A

    Parameters
    ----------
    A : (N, N) array-like
        Undirected weighted adjacency matrix
    rewiring_iter : int, optional
        Rewiring parameter (each edge is rewired approximately maxswap times).
        Default = 10.
    sort_freq : float, optional
        Frequency of weight sorting. Must be between 0 and 1.
        If 1, weights are sorted at every iteration.
        If 0.1, weights are sorted at every 10th iteration.
        A higher value results in a more accurate strength sequence match.
        Default = 1.
    R : (N, N) array-like, optional
        Pre-randomized adjacency matrix.
        If None, a rewired adjacency matrix is generated using the
        Maslov & Sneppen algorithm.
        Default = None.
    connected: bool, optional
        Whether to ensure connectedness of the randomized network.
        By default, this is inferred from data.
    seed: float, optional
        Random seed. Default = None.
    
    Returns
    -------
    B : (N, N) array-like
        Randomized adjacency matrix

    Notes
    -------
    Uses Maslov & Sneppen rewiring to produce a
    surrogate adjacency matrix, B, with the same 
    size, density, and degree sequence as A. 
    The weights are then permuted to optimize the
    match between the strength sequences of A and B.

    This function is adapted from a function written in MATLAB 
    by Mika Rubinov (https://sites.google.com/site/bctnet/home).
    It was adapted to positive structural connectivity networks 
    from an algorithm originally developed for 
    signed functional connectivity networks.

    References
    -------
    Maslov & Sneppen (2002) Specificity and stability in 
    topology of protein networks. Science.
    Rubinov & Sporns (2011) Weight-conserving characterization of
    complex functional brain networks. Neuroimage.
    """

    A = A.copy()
    try:
        A = np.asarray(A)
    except TypeError as err:
        msg = ('A must be array_like. Received: {}.'.format(type(A)))
        raise TypeError(msg) from err
    
    if sort_freq > 1 or sort_freq <= 0:
        msg = ('sort_freq must be between 0 and 1. '
               'Received: {}.'.format(sort_freq))
        raise ValueError(msg)

    rs = check_random_state(seed)

    n = A.shape[0]

    #clearing diagonal
    np.fill_diagonal(A, 0)

    if R is None:
        #ensuring connectedness if the original network is connected
        if connected is None:
            connected = False if bct.number_of_components(A) > 1 else True

        #Maslov & Sneppen rewiring
        if connected:
            R = bct.randmio_und_connected(A, rewiring_iter, seed = seed)[0]
        else:
            R = bct.randmio_und(A, rewiring_iter, seed = seed)[0]

    B = np.zeros((n, n))
    s = np.sum(A, axis = 1) #strengths of A
    sortAvec = np.sort(A[np.triu(A, k=1) > 0]) #sorted weights vector
    x, y = np.nonzero(np.triu(R, k=1)) #weights indices

    E = np.outer(s, s) #expected weights matrix

    if sort_freq == 1:
        for i in range(len(sortAvec) -1, -1, -1):
            sort_idx = np.argsort(E[x, y]) #indices of x and y that sort E

            r = math.ceil(rs.rand()*i)
            r_idx = sort_idx[r] #random index of sorted expected weight matrix

            #assigning corresponding sorted weight at this index
            B[x[r_idx], y[r_idx]] = sortAvec[r]

            #radjusting the expected weight probabilities of
            #the node indexed in x
            f = 1 - sortAvec[r]/s[x[r_idx]]
            E[x[r_idx], :] *= f
            E[:, x[r_idx]] *= f

            #radjusting the expected weight probabilities of
            #the node indexed in y
            f = 1 - sortAvec[r]/s[y[r_idx]]
            E[y[r_idx], :] *= f
            E[:, y[r_idx]] *= f

            #readjusting residual strengths of nodes indexed in x and y
            s[x[r_idx]] -= sortAvec[r]
            s[y[r_idx]] -= sortAvec[r]

            #removing current weight
            x = np.delete(x, r_idx)
            y = np.delete(y, r_idx)
            sortAvec = np.delete(sortAvec, r)
    else:
        sort_period = round(1/sort_freq) #sorting period
        for i in range(len(sortAvec) -1, -1, -sort_period):
            sort_idx = np.argsort(E[x, y]) #indices of x and y that sort E

            r = rs.choice(i, min(i, sort_period), replace = False)
            r_idx = sort_idx[r]#random indices of sorted expected weight matrix

            #assigning corresponding sorted weights at these indices
            B[x[r_idx], y[r_idx]] = sortAvec[r]

            xy_nodes = np.append(x[r_idx], y[r_idx]) #randomly indexed nodes
            xy_nodes_idx = np.unique(xy_nodes) #randomly indexed nodes' indices

            #nodal cumulative weights
            accumWvec = np.bincount(xy_nodes, 
                                    weights = sortAvec[np.append(r, r)],
                                    minlength = n)

            #readjusting expected weight probabilities
            F = 1 - accumWvec[xy_nodes_idx]/s[xy_nodes_idx]
            F = F[:, np.newaxis]

            E[xy_nodes_idx, :] *= F
            E[:, xy_nodes_idx] *= F.T

            #readjusting residual strengths of nodes indexed in x and y
            s[xy_nodes_idx] -= accumWvec[xy_nodes_idx]

            #removing current weight
            x = np.delete(x, r_idx)
            y = np.delete(y, r_idx)
            sortAvec = np.delete(sortAvec, r)

    B += B.T

    return B