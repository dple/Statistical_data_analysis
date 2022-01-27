# Statistical Data Analysis

This project implements some common statistical data analysis methods, including:


#### 1. T-test: 
Given two samples, perform t-test to determine if there is a significant difference between the means of these two samples.
This test is mostly used when the data sets follow a normal distribution.

#### 2. Mann-Whitney U test
Mann-Whitney U test is used to perform two samples hypothesis tests if their distributions are unknown.
Formula:
           t = (mean1 - mean2)/sqrt(sem1^2 - sem2^2)
           mean1, mean2: means of two samples
           sem1, sem2: standard error of two samples
           
#### 3. Kullback–Leibler divergence (KLD), 
KLD is also called relative entropy

#### 4. Empirical Cumulative Distribution Functions (CDF) distance
Empirical CDF distance of two non-parametric variables. 

Formula:  
    D = sum((CDF(X) - CDF(Y))**2)

#### 5. Entropy
Calculate entropy by binning. The number of bins is determined by different rules depending on the variable's distributions

#### 6. Kolmogorov–Smirnov (KS) Test
KS test that can be used to compare two non-parametric variables. 

Formula:  
    D = max(abs(CDF(X) - CDF(Y)))

#### 7. Mutual Information
Calculate the mutual information between X and Y. Using single and joint entropies:
https://en.wikipedia.org/wiki/Mutual_information#Relation_to_conditional_and_joint_entropy

Formula: 
    I(X, Y) = H(X) + H(Y) - H(X, Y)
