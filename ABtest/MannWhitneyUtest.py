"""
Project data statistical analysis
Author: Le, Duc Phong
Date: Sep 19, 2020
"""
import numpy as np
from scipy.stats import mannwhitneyu

'''
Mann-Whitney U test is used to perform two samples hypothesis tests if their distributions are unknown.

Formula:
           t = (mean1 - mean2)/sqrt(sem1^2 - sem2^2)
           mean1, mean2: means of two samples
           sem1, sem2: standard error of two samples
'''

def mann_whitneyu(data1, data2):
    # compare samples
    stat, p = mannwhitneyu(data1, data2,alternative='two-sided')
    return stat, p

if __name__ == '__main__':
    x = np.random.randint(0,9999,(500,1))
    y = np.random.randint(1000, 10000, (1000,1))

    alpha = 0.05
    u_stat, p = mann_whitneyu(x, y)
    if p > alpha:
        print('Same distributions (fail to reject H0)')

    else:
        print('Different distributions (reject H0)')
