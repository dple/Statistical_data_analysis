"""
Calculate entropy using different methods

Phong D. Le  -  le.duc.phong@gmail.com
"""

import numpy as np
from numpy.random import seed, randn, normal
import math
from scipy.stats import iqr
from statistics import stdev

EPS = np.finfo(float).eps

def determine_nbins1D(X, rule):
    '''
    There are three common methods to determine the number of bins to compute entropy of ONE variable X
    :param X: array-like of numbers
    :param rule:    1) Freedman‐Diaconis's rule: used for unknown distributions or non-parametric
                            nbins = ceil(max(X) - min(X) / 2 * IQR * N^{-1/3})
                    2) Scotts's Rule: used for normal distribution
                            nbins = ceil(max(X) - min(X) / 3.5 * STD * N^{-1/3})
                    3) Sturges' Rule
                            nbins = ceil(1 + log2(n))

    :return: the optimal number of bins used to calculate entropy
    '''
    maxmin_range = max(X) - min(X)
    n = len(X)
    n3= n ** (-1/3)
    if rule == 'Freedman‐Diaconis':
        return math.ceil(maxmin_range/(2.0 * iqr(X) * n3))
    if rule == 'Scott':
        return math.ceil(maxmin_range / (3.5 * stdev(X) * n3))
    if rule == 'Sturges':
        return math.ceil(1 + math.log(n, 2))
    return 0


def single_entropy(X, dist):
    """
    Calculate the entropy of a random variable X
    """
    rule = 'Sturges'
    if dist == 'normal':
        rule = 'Scott'
    elif dist == 'unknown':
        rule = 'Freedman‐Diaconis'

    nbins = determine_nbins1D(X, rule)
    p, _ = np.histogram(X, nbins)
    p = p / p.sum()

    HX = 0.0
    for i in range(len(p)):
        if p[i] > 0:   # p[i] += EPS     # Add a small value EPS if the probability of a bin equals to 0
            HX += p[i] * math.log2(p[i])

    return -HX

def determine_nbins2D(X1, X2, rule1, rule2):
    nbins1 = determine_nbins1D(X1, rule1)
    nbins2 = determine_nbins1D(X2, rule2)

    return math.ceil((nbins1 + nbins2)/2)


def joint_entropy(X1, X2, dist1, dist2):
    """
    Calculate the joint entropy of two variables X1, and X2
    """
    rule1 = 'Sturges'
    if dist1 == 'normal':
        rule1 = 'Scott'
    elif dist1 == 'unknown':
        rule1 = 'Freedman‐Diaconis'

    rule2 = 'Sturges'
    if dist2 == 'normal':
        rule2 = 'Scott'
    elif dist2 == 'unknown':
        rule2 = 'Freedman‐Diaconis'

    nbins = determine_nbins2D(X1, X2, rule1, rule2)
    jointProbs, _, _ = np.histogram2d(X1, X2, bins=nbins)
    jointProbs = jointProbs / jointProbs.sum()

    HXY = 0
    rows = len(jointProbs)
    cols = len(jointProbs[0])

    for i in range(rows):
        for j in range(cols):
            if jointProbs[i][j] > 0:    #        jointProbs[i][j] += EPS
                HXY += jointProbs[i][j] * math.log2(jointProbs[i][j])

    return -HXY

# seed random number generator
seed(1)
# prepare data
X1 = 20 * randn(1000) + 100
X2 = 10 * normal(0, 20, 1000)
nbins = determine_nbins1D(X1, rule='Freedman‐Diaconis')
print('No bins for X1 = {}'.format(nbins))

nbins = determine_nbins2D(X1, X2, 'Freedman‐Diaconis', 'Scott')
print('No bins for X1 and X2 = {}'.format(nbins))

entropy = single_entropy(X1, 'unknown')
print('Entropy of X1 = {}'.format(entropy))

joint_entropy = joint_entropy(X1, X2, 'unknown', 'normal')
print('Joint Entropy of X1 and X2 = {}'.format(joint_entropy))