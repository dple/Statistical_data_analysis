"""
Implement Kolmogorov–Smirnov test that can be used to compare two non-parametric variables
https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test

Phong D. Le  -  le.duc.phong@gmail.com
"""

import numpy as np
from scipy.stats import iqr, tstd
def KSTest(X, Y):
    """
    Calculate Kolmogorov–Smirnov statistic of two samples

    Formula:  D = max(abs(CDF(X) - CDF(Y)))
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

    return max(abs(cdfX - cdfY))


if __name__ == '__main__':
    # X, a random Gaussian distribution
    X = np.random.normal(0, 100, 1000)
    # Y, a random bivariate Gaussian distribution
    Y = np.concatenate([np.random.normal(0, 50, 500),
                        np.random.normal(200, 70, 500)])

    print('Kolmogorov–Smirnov test: D(X,Y) = {}'.format(KSTest(X, Y)))
