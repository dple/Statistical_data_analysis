"""
Calculate entropy using different methods

Phong D. Le  -  le.duc.phong@gmail.com
"""

import numpy as np
import math
from entropy import determine_nbins2D
from numpy.random import seed, randn, normal

EPS = np.finfo(float).eps

def KL_divergence(X1, X2, nbins):
    amax = max(np.max(X1), np.max(X2))
    amin = min(np.min(X1), np.min(X2))
    P, _ = np.histogram(X1, bins=nbins, range=[amin, amax])
    Q, _ = np.histogram(X2, bins=nbins, range=[amin, amax])
    P = P / P.sum()
    Q = Q / Q.sum()

    kld = 0
    for i in range(len(P)):
        if (P[i] > 0) and (Q[i] > 0):
            kld += P[i] * math.log2(P[i]/Q[i])

    return kld


# seed random number generator
seed(1)
# prepare data
X1 = 20 * randn(1000) + 100
X2 = 10 * normal(0, 20, 1000)

nbins = determine_nbins2D(X1, X2, 'Freedman‐Diaconis', 'Scott')
print('No bins for X1 and X2 = {}'.format(nbins))

kld = KL_divergence(X1, X2, nbins)
print('Kullback-Leibler Divergence of X1 and X2 = {}'.format(kld))