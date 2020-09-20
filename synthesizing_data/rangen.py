"""
Project data statistical analysis
Author: Le, Duc Phong
Date: Sep 19, 2020
"""

"""
Project data statistical analysis
Author: Le, Duc Phong
Date: Sep 19, 2020
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns

def bivariate_normal_dist():
    # Make up some random data
    x = np.concatenate([np.random.normal(20, 20, 2500),
                        np.random.normal(100, 20, 2500)])
    return x


def multivariate_normal_dist():
    # Make up some random data
    x = np.concatenate([np.random.normal(0, 20, 1000),
                        np.random.normal(100, 30, 2500),
                        np.random.normal(300, 40, 2500),
                        np.random.normal(500, 30, 1000)])

    return x


def nonlinear_corr_dist():
    x = np.concatenate([np.random.normal(0, 1, 5000),
                        np.random.normal(4, 1, 5000)])

    y = x * x
    df = pd.DataFrame(data={'col1': x, 'col2': y})
    df = shuffle(df)
    return df


def linear_corr_dist():
    x = np.concatenate([np.random.normal(0, 10, 5000),
                        np.random.normal(50, 10, 5000)])

    ecdf = ECDF(x)
    inv_cdf = interp1d(ecdf.y, ecdf.x, bounds_error=False, assume_sorted=True)
    r = np.random.uniform(0, 1, 10000)
    y = inv_cdf(r)

    z = 2 * x + 3 * y
    df = pd.DataFrame(data={'col1': x, 'col2': y, 'combination': z})
    df = shuffle(df)
    return df


#sns.distplot(bivariate_normal_dist())
#sns.distplot(multivariate_normal_dist())
#sns.distplot(nonlinear_corr_dist()['col1'])
#sns.distplot(nonlinear_corr_dist()['col2'])
sns.distplot(linear_corr_dist()['col1'])
sns.distplot(linear_corr_dist()['col2'])
sns.distplot(linear_corr_dist()['combination'])
plt.show()
