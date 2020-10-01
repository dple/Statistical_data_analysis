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


def visualize2D(X):
    '''
    Show histogram of the variable X
    '''

    # Calculate the number of bins using Sturges' rule
    # Formula:    bins = ceil(1 + log2(n))
    bins = np.ceil(1 + np.log2(len(X))).astype('int')
    plt.hist(X, bins=bins, density=True)

def visualize3D(X, Y):
    '''
    Show 3D histogram of joint probabily of X and Y
    '''
    # Determine the number of bins using Sturges' rule
    binX = np.ceil(1 + np.log2(len(X))).astype('int')
    binY = np.ceil(1 + np.log2(len(Y))).astype('int')

    # Create 2d histogram
    Hist, _, _ = np.histogram2d(X, Y, bins=[binX, binY])

    # Define 3d axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create an X-Y mesh of the 2D data
    x_data, y_data = np.meshgrid(np.arange(Hist.shape[1]), np.arange(Hist.shape[0]))

    # Flatten out the arrays to pass to bar3d
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = Hist.flatten()
    ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data)


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

    # Plot 2D histogram of the variable
    visualize2D(X)
    visualize2D(Y)
    plt.show()

    # Plot 3D histogram of the two variables
    visualize3D(X, Y)
    plt.show()
