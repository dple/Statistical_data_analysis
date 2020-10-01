"""
Calculate mutual information between two variables
https://en.wikipedia.org/wiki/Mutual_information

Phong D. Le  -  le.duc.phong@gmail.com
"""

import numpy as np
from entropy import single_entropy, joint_entropy
from numpy.random import seed, randn, normal

def mutual_information(X, Y, dist1=None, dist2 = None):
    """
    Calculate the mutual information between X and Y. Using single and joint entropies:
    https://en.wikipedia.org/wiki/Mutual_information#Relation_to_conditional_and_joint_entropy

    I(X, Y) = H(X) + H(Y) - H(X, Y)
    """

    HX = single_entropy(X, dist1)
    HY = single_entropy(Y, dist2)
    HXY = joint_entropy(X, Y, dist1, dist2)
    mi = HX + HY - HXY

    return mi
    
if __name__ == '__main__':

    # seed random number generator
    seed(1)
    # prepare data
    X = 20 * np.random.randn(1000) + 100
    Y = X + (10 * np.random.randn(1000) + 50)
    
    mi = mutual_information(X, Y, 'unknown', 'normal')
    print('Mutual Information between X and Y uing entropies, I(X, Y) = {}'.format(mi))
