"""
Show statistical information 

Le, Phong D.  -  le.duc.phong@gmail.com
Date: Sep 19, 2020
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr, pearsonr, spearmanr, kendalltau


def statshow(X):
    """
    Using scipy.stats to show basic statistical information of variables
    1. Mean
    2: Standard Deviation
    3. Median
    4. Variance
    5. IQR
    """
    print('Basic statistics of X:\n Mean = {}, \n '
          'Standard Deviation = {},\n '
          'Median = {},\n Variance = {}, '
          '\n IQR = {}.\n'.
          format(np.nanmean(X), np.nanstd(X), np.nanmedian(X), np.nanvar(X), iqr(X)))

def corshow(X, Y):
    """
    Show the correlation of the two variable. There are three types of correlation measures
    1. Pearson correlation measures the linear relationship between two variables
    2. Spearman and Kendall compare the ranks of data in two variables
    """
    print('Pearson Correlation between X and Y is \n{}'.format(pearsonr(X, Y)[0]))
    print('Spearman Correlation between X and Y is \n{}'.format(spearmanr(X, Y)[0]))
    print('Kendall tau Correlation between X and Y is \n{}'.format(kendalltau(X, Y)[0]))



if __name__ == '__main__':
    # X, a random Gaussian distribution
    X = np.random.normal(0, 100, 1000)
    # Y, a random bivariate Gaussian distribution
    Y = np.concatenate([np.random.normal(0, 50, 500),
                        np.random.normal(200, 70, 500)])


    # Show stats information of X and Y
    statshow(X)
    statshow(Y)

    # Show correlation information between X and Y
    corshow(X, Y)

    # Plot histogram of the variable
    plt.hist(X)
    plt.hist(Y)
    plt.show()
