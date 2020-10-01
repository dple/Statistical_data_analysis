"""
Calculate mutual information between two variables

Phong D. Le  -  le.duc.phong@gmail.com
"""

import numpy as np
from numpy.random import seed, rand


def mutual_information(X, Y):
    '''
    Calculate mutual information based on 2D histrograms

    The optimal number of bins for MI is determined by the following paper

    Abdenour Hacine-Gharbia, Philippe Ravier, "A Binning Formula of Bi-histogram
    for Joint Entropy Estimation Using Mean Square Error Minimization",
    Pattern Recognition Letters, Volume 101, 1 January 2018, Pages 21-28
    https://www.sciencedirect.com/science/article/pii/S0167865517304142

    k = round(1/sqrt(2) * sqrt(1 + sqrt(1 + 24*N/(1 - rho**2))))
    '''
    rho = np.corrcoef(X, Y)[0][1]
    N = max(len(X), len(Y))
    bins = round(1 / np.sqrt(2) * np.sqrt(1 + np.sqrt(1 + 24 * N / (1 - rho * rho))))

    pxy, _, _ = np.histogram2d(X, Y, bins)
    pxy = pxy / pxy.sum()

    px, _ = np.histogram(X, bins)
    px = px / px.sum()

    py, _ = np.histogram(Y, bins)
    py = py / py.sum()

    probs = pxy / (np.reshape(px, [-1, 1]) * py)

    return (pxy * np.ma.log2(probs)).sum()


if __name__ == '__main__':

    # seed random number generator
    seed(1)
    # prepare data
    X = rand(10000)
    Y = rand(10000)

    print('Mutual Information between X and Y uing bins, I(X, Y) = {}'.format(mutual_information(X, Y)))
