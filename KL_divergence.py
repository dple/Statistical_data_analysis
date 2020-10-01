"""
Calculate Kullback-Leibler divergence (KLD). KLD is also called relative entropy
https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

Phong D. Le  -  le.duc.phong@gmail.com
"""

import numpy as np
from entropy import determine_nbins2D
from numpy.random import seed, randn, normal

EPS = np.finfo(float).eps

def KL_divergence(X, Y, nbins):
    """
    Calculate Kullback-Leibler divergence of X and Y

    Formula: KLD(X, Y) = D_KL (Px || Py) = sum(px * log2 (px / py))
    """
    amax = max(np.max(X), np.max(Y))
    amin = min(np.min(X), np.min(Y))
    p, _ = np.histogram(X, bins=nbins, range=[amin, amax])
    q, _ = np.histogram(Y, bins=nbins, range=[amin, amax])
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
    X = 20 * randn(1000) + 100
    Y = 10 * normal(0, 20, 1000)

    # Use Freedman‐Diaconis' rule if the distribution is unknown.
    # If the distribution is normal, using Scott's rule
    nbins = determine_nbins2D(X, Y, 'Freedman‐Diaconis', 'Scott')

    print('Kullback-Leibler Divergence of X over Y = {}'.format(KL_divergence(X, Y, nbins)))

    # KL divergence is not symmetric, that is KLD(X, Y) <> KLD (Y, X)
    print('Kullback-Leibler Divergence of Y over X = {}'.format(KL_divergence(Y, X, nbins)))
