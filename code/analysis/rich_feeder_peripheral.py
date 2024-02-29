import numpy as np
from scipy.stats import ttest_ind

try:
    from numba import njit
    use_numba = True
except ImportError:
    use_numba = False

def _binarize(W):
    """
    Binarizes a matrix
    Parameters
    ----------
    W : (N, N) array_like
        Matrix to be binarized
    Returns
    -------
    binarized : (N, N) numpy.ndarray
        Binarized matrix
    """
    return (W > 0) * 1

if use_numba:
    _binarize = njit(_binarize)

def degrees_und(W):
    """
    Computes the degree of each node in `W`
    Parameters
    ----------
    W : (N, N) array_like
        Unweighted, undirected connection weight array.
        Weighted array will be binarized prior to calculation.
        Directedness will be ignored (out degree / row sum taken).
    Returns
    -------
    deg : (N,) numpy.ndarray
        Degree of each node in `W`
    """
    return np.sum(_binarize(W), axis=0)

#added sum as a statistic
#added return of number of rich links
def rich_feeder_peripheral(x, sc, stat='median'):
    """
    Calculates connectivity values in rich, feeder, and peripheral edges.

    Parameters
    ----------
    x : (N, N) numpy.ndarray
        Symmetric correlation or connectivity matrix
    sc : (N, N) numpy.ndarray
        Binary structural connectivity matrix
    stat : {'mean', 'median', 'sum'}, optional
        Statistic to use over rich/feeder/peripheral links. Default: 'median'

    Returns
    -------
    rfp : (3, k) numpy.ndarray
        Array of median rich (0), feeder (1), and peripheral (2)
        values, defined by `x`. `k` is the maximum degree defined on `sc`.
    pvals : (3, k) numpy.ndarray
        p-value for each link, computed using Welch's t-test.
        Rich links are compared against non-rich links. Feeder links are
        compared against peripheral links. Peripheral links are compared
        against feeder links. T-test is one-sided.
    rc_n : (1, k) numpy.ndarray
        Number of rich links at each degree threshold.
    
    Notes
    -----
    This code was written by Justine Hansen who promises to fix and even
    optimize the code should any issues arise, provided you let her know.
    """
    stats = ['mean', 'median', 'sum']
    if stat not in stats:
        raise ValueError(f'Provided stat {stat} not valid.\
                         Must be one of {stats}')
    nnodes = len(sc)
    mask = np.triu(np.ones(nnodes), 1) > 0
    node_degree = degrees_und(sc)
    k = np.max(node_degree).astype(np.int64) + 1
    rfp_label = np.zeros((len(sc[mask]), k))

    for degthresh in range(k):  # for each degree threshold
        hub_idx = np.where(node_degree >= degthresh)  # find the hubs
        hub = np.zeros([nnodes, 1])
        hub[hub_idx, :] = 1
        rfp = np.zeros([nnodes, nnodes])  # for each link, define rfp
        for edge1 in range(nnodes):
            for edge2 in range(nnodes):
                if hub[edge1] + hub[edge2] == 2:
                    rfp[edge1, edge2] = 1  # rich
                if hub[edge1] + hub[edge2] == 1:
                    rfp[edge1, edge2] = 2  # feeder
                if hub[edge1] + hub[edge2] == 0:
                    rfp[edge1, edge2] = 3  # peripheral
        rfp_label[:, degthresh] = rfp[mask]

    rfp = np.zeros([3, k])
    pvals = np.zeros([3, k])
    rc_n = np.zeros([1, k])

    for degthresh in range(k):
        if stat == 'median':
            redfunc = np.median
        elif stat == 'sum':
            redfunc = np.sum
        else:
            redfunc = np.mean
        for linktype in range(3):
            if linktype == 0:
                rc_n[:, degthresh] = len(x[mask][rfp_label[:, degthresh] ==
                                                 linktype + 1])
            rfp[linktype, degthresh] = redfunc(x[mask][rfp_label[:, degthresh]
                                                       == linktype + 1])
        # p-value (one-sided Welch's t-test)
        _, pvals[0, degthresh] = ttest_ind(
            x[mask][rfp_label[:, degthresh] == 1],
            x[mask][rfp_label[:, degthresh] != 1],
            equal_var=False, alternative='greater')
        _, pvals[1, degthresh] = ttest_ind(
            x[mask][rfp_label[:, degthresh] == 2],
            x[mask][rfp_label[:, degthresh] == 3],
            equal_var=False, alternative='greater')
        _, pvals[2, degthresh] = ttest_ind(
            x[mask][rfp_label[:, degthresh] == 3],
            x[mask][rfp_label[:, degthresh] == 2],
            equal_var=False, alternative='greater')

    return rfp, pvals, rc_n
