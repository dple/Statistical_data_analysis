"""
Empirical Cumulative Distribution Functions distance of two non-parametric variables

Phong D. Le  -  le.duc.phong@gmail.com
"""

import numpy as np

def eCDF(X, Y):
    """
    Calculate eCDF of two samples

    Formula:  D = sum((CDF(X) - CDF(Y))**2)
    """
    # Calculate the number of bins using Sturges' rule
    # Formula:    bins = ceil(1 + log2(n))
    bins1 = np.ceil(1 + np.log2(len(X))).astype('int')
    bins2 = np.ceil(1 + np.log2(len(Y))).astype('int')
    # Number of bins will be the average of two bins
    bins = np.ceil((bins1 + bins2)/2).astype('int')

    # Split data of two variables in the same range and the same number of bins
    amax = max(np.max(X), np.max(Y))
    amin = min(np.min(X), np.min(Y))
    histX, _ = np.histogram(X, bins=bins, range=[amin, amax])
    histY, _ = np.histogram(Y, bins=bins, range=[amin, amax])
    # Get the probability of bins
    px = histX / histX.sum()
    py = histY / histY.sum()
    # Get cumulative distribution probability
    cdfX = np.cumsum(px)
    cdfY = np.cumsum(py)

    return ((cdfX - cdfY)**2).sum()


if __name__ == '__main__':
    # X, a random Gaussian distribution
    X = np.random.normal(0, 100, 1000)
    # Y, a random bivariate Gaussian distribution
    Y = np.concatenate([np.random.normal(0, 50, 500),
                        np.random.normal(200, 70, 500)])

    print('Empirical Cumulative Distribution Functions distance: D(X,Y) = {}'.format(eCDF(X, Y)))