"""
Project data statistical analysis
Author: Le, Duc Phong
Date: Sep 19, 2020
"""
import numpy as np
from scipy.stats import sem, t

'''
Given two samples, perform t-test to determine if there is a significant difference between the means of these two samples.
This test is mostly used when the data sets follow a normal distribution.
'''

# Calculating the t-test for two samples
# Formula:
#           t = (mean1 - mean2)/sqrt(sem1^2 - sem2^2)
#           mean1, mean2: means of two samples
#           sem1, sem2: standard error of two samples

def ttest(data1, data2, alpha):
    mean1, mean2 = np.mean(data1), np.mean(data2)
    se1, se2 = sem(data1), sem(data2)
    sed = np.sqrt(se1**2 + se2**2)
    t_stat = (mean1 - mean2) / sed # t-statistic
    df = len(data1) + len(data2)   # degree of freedom
    cv = t.ppf(1.0 - alpha, df)    # critical value, used to determine if two variables are same or different
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0    # p-value. If p > alpha then two variable are the same with the confidence of 1 - alpha
    if p < alpha:
        print('Two varibles have different mean')
    else:
        print('Two variables have the same mean')

    return t_stat, df, cv, p


if __name__ == '__main__':
    x = np.random.normal(0, 30, 500)
    y = np.random.normal(10, 100, 1000)
    alpha = 0.05
    ttest(x, y, alpha)
