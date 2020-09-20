"""
Project data statistical analysis
Author: Le, Duc Phong
Date: Sep 19, 2020  
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stats
import scipy.stats

# X, a random Gaussian distribution
X = np.random.normal(0, 100, 1000)
# Y, a random bivariate Gaussian distribution
Y = np.concatenate([np.random.normal(0, 50, 500),
                    np.random.normal(200, 70, 500)])

# Plot histogram of the variable
sns.distplot(X)
sns.distplot(Y)
plt.show()


'''
Show basic statistics of variables. Using package Statistics to show basic statistical information of variables
1. Mean: mean of data
2. Median: middle value of numeric data
3. Variance: sample variance of data
4: Standard Deviation: square root of variance
'''
print('Basic statistics of X:\n Mean = {},\n Median = {},\n Variance = {},\n Standard Deviation = {}.\n'.
      format(stats.mean(X), stats.median(X), stats.variance(X), stats.stdev(X)))
print('Basic statistics of Y:\n Mean = {},\n Median = {},\n Variance = {},\n Standard Deviation = {}.\n'.
      format(stats.mean(Y), stats.median(Y), stats.variance(Y), stats.stdev(Y)))

'''
Show the correlation of the two variable. There are three types of correlation measures
1. Pearson correlation measures the linear relationship between two variables
2. Spearman and Kendall compare the ranks of data in two variables
'''
print('Pearson Correlation between X and Y is \n{}'.format(scipy.stats.pearsonr(X, Y)[0]))
print('Spearman Correlation between X and Y is \n{}'.format(scipy.stats.spearmanr(X, Y)[0]))
print('Kendall tau Correlation between X and Y is \n{}'.format(scipy.stats.kendalltau(X, Y)[0]))
