"""
LEARNING NOTES 
- based on following tutorial: https://realpython.com/python-statistics/
- using np, pd, math, scipy.stats to calculate simple statistics measures like, mean, std etc..
"""

import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd

x = [8.0, 1, 2.5, 4, 28.0]
x_w_nan = [8.0, 1, 2.5, math.nan,  4, 28.0]

np.nan

y, ynan = np.array(x), np.array(x_w_nan)
z, znan = pd.Series(x), pd.Series(x_w_nan)

y
z
x

m = sum(x)/len(x)
m = statistics.mean(x)

# two ways how to calcualte mean with numpy, one with a function and one with a method
np.mean(y) # floating point number mean
y.mean()

ynan.mean()
np.nanmean(ynan) #mean ignoring nan
ynan.nanmean() #but this method does not exist as opposed to normal mean()

znan.mean()  #pandas ignores nan by default X numpy (previous line)

x
w = [0.1, 0.5, 0.1, 0.1, 0.2]

wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)
wmean

wmean = np.average(y, weights = w)
wmean

sum(w * y) 

#harmonic mean
len(x) / sum(1 / i for i in x)
statistics.harmonic_mean(ynan) #does not work with nan
scipy.stats.hmean(ynan) #does not work with nan

gmean = 1
for i in x:
    gmean *= i  
gmean **= 1/len(x)    

scipy.stats.gmean(x)

statistics.median(x)
sorted(x)    
statistics.mode(x) # there has to be a unique mode, remedy found in the line below
scipy.stats.mode(x)

statistics.variance(x)

np.var(x, ddof=1) #degrees of freedom (n-1)
np.var(x) #var varies based on DoF!!!!!!!!!
np.asarray(x).var()

statistics.stdev(x)
np.std(x, ddof = 1)
np.std(x) #std varies based on DoF!!!!!!!!!
np.std?

statistics.stdev(x)
np.percentile(x, 95)

scipy.stats.describe(x)

np.cov(x, x)
np.corrcoef(x, x)

scipy.stats.linregress(x, y)

v = pd.Series(x)
u = pd.Series(y)
v.corr(u)

## -- Working With 2D Data
a = np.array([[1,1,1,],
             [2,3,1],
             [4,9,2],
             [8,27,4],
             [16,1,1]]
             )

np.mean(a)
a.mean()
a.var(ddof=1)

np.mean(a, axis = 1)
np.mean(a, axis = None)
np.mean(a, axis = 0)
a.mean(axis = 0)  #for each column
scipy.stats.gmean(a, axis = 0)
