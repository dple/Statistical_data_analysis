"""
Project data statistical analysis

Given a dataset, generate more columns from the exiting columns

Author: Le, Duc Phong
Date: Sep 19, 2020
"""

import pandas as pd
from random import seed, random

# Seed random number generator
df = pd.read_csv('thyroid.csv')

# Fabonacci numbers
ncols = [1, 1]
for i in range(2, 16):
    x = ncols[i - 2] + ncols[i - 1]
    ncols += [x]

cols = df.columns

for n in ncols:
    seed(n)
    fname = 'thyroid' + str(n) + '.csv'
# Linear combination
    for i in range(n//2):
        name = 'synt' + str(i)
        df[name] = 0
        for col in cols:

            r = random()
            if (r > .25) and (r < .5):
                df[name] = df[name] + (r * df[col])
            if (r >= .5) and (r < .75):
                df[name] = abs(df[name] - (r * df[col]))
# Linear combination
    for i in range(n//2, n, 1):
        name = 'synt' + str(i)
        df[name] = 0
        for col in cols:
            r = random()
            if (r >= .5) and (r < .75):
                df[name] = df[name] + (r * df[col] * df[col])
            if (r > .25) and (r < .5):
                df[name] = abs(df[name] - (r * df[col] * df[col]))

            r = random()
            if (r < .25):
                df[name] = df[name] + (r * df[col])
            if (r >= .75):
                df[name] = abs(df[name] - (r * df[col]))

    df.to_csv(fname, index=False)

