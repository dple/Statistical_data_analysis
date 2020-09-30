"""
Calculate Kullback-Leibler divergence (KLD). KLD is also called relative entropy
https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

Phong D. Le  -  le.duc.phong@gmail.com
"""

import numpy as np
from entropy import determine_nbins2D
from numpy.random import seed, randn, normal

EPS = np.finfo(float).eps

def KL_divergence(X1, X2, nbins):
    """
    Calculate Kullback-Leibler divergence of X1 and X2
    """
    amax = max(np.max(X1), np.max(X2))
    amin = min(np.min(X1), np.min(X2))
    p, _ = np.histogram(X1, bins=nbins, range=[amin, amax])
    q, _ = np.histogram(X2, bins=nbins, range=[amin, amax])
    p = p / p.sum()
    q = q / q.sum()

    kld = 0

    for i in range(len(p)):
        if (p[i] > 0) and (q[i] > 0):
            kld += p[i] * np.log2(p[i]/q[i])
    return kld

if __name__ == '__main__':

    # seed random number generator
    seed(1)
    # prepare data
    X1 = 20 * randn(1000) + 100
    X2 = 10 * normal(0, 20, 1000)

    # Use Freedman‐Diaconis' rule if the distribution is unknown. 
    # If the distribution is normal, using Scott's rule
    nbins = determine_nbins2D(X1, X2, 'Freedman‐Diaconis', 'Scott')
    print('No bins for X1 and X2 = {}'.format(nbins))

    kld = KL_divergence(X1, X2, nbins)
    print('Kullback-Leibler Divergence of X1 and X2 = {}'.format(kld))
