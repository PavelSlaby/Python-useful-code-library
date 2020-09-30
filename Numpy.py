## Arrays
v = [0.5, 0.75, 1.0, 1.5, 2.0]
w = [5, 75, 1, 1, 2]
u = [3, 4, 5, 6, 7]
t = [0, 0, 0, 0, 0]

m = [v, w, u, t]
m
print(m)

m[1]
m[1][1]

#there is python array thingy
import array
array.array('f',[1, 2, 3]) 
#this is not very convenient

import numpy as np
np.__version__

a = np.array([0, 0.5, 10 , 0])
type(a)
a.dtype

a[0]
a[-1]
a[1::2]
a[::2]
a[1:4:2]
a[:2]
a * 2
a.sum()
np.sum(a)
a.std()
a.cumsum()
np.sqrt(a)
np.max(a)
np.min(a)
np.std(a)
np.mean(a)
a.argmin()
a.argmax()

b = np.array([a, a * 2])
print(a)
print(b)

b[0][1]
b[1][2]
b[1, 2] #same as b[1][2]

b.sum(axis=0) #sum columnwise - same as pundas
b.sum(axis=1) #sum rownwise - sae as pandas

u = np.array([0, 0.5, 10 , 0], dtype = 'S') #string type
print(u)
u = u.astype('f') # change type

d = np.zeros((5, 2, 3))
d

e = np.ones([2,2])
e
a = list(e)

d3 = np.array([[[1, 2], [1, 2]], [[3, 4], [3, 4]], [[3, 4], [3, 4]], [[3, 4], [3, 4]], [[3, 4], [3, 4]]])
d3
d3.ndim

nd = np.array([1 , 2, 3 , 4 ], ndmin = 5)
'''
In this array the innermost dimension (5th dim) has 4 elements, 
the 4th dim has 1 element that is the vector, 
the 3rd dim has 1 element that is the matrix with the vector, 
the 2nd dim has 1 element that is 3D array and 1st dim has 1 element that is a 4D array.
'''
print(nd)
nd

a = np.array([1, 2, 4, 3])
w = a.copy()
c = a.view()

a[0]= 100 
a
c # c is a view - has changed
w # w is a copy - did not change
print(c.base) #view returns the original array
print(w.base) #the copy returns none

np.pi

ar1 = np.array([[1, 2, 3], [10, 20, 30]])
ar2 = np.array([[4, 5, 6], ['a', 20, 30]])

ar = np.concatenate((ar1, ar2))
ar
ar = np.concatenate((ar1, ar2), axis = 1) #equivalent to hstack
ar
ar = np.concatenate((ar1, ar2), axis = 0) #equivalent to vstack
ar

br = np.hstack((ar1, ar2))
br
br = np.vstack((ar1, ar2))
br

br[-1,::2]  # This is a slice of the last row, and only every other element.

cr  = np.array_split(br, 4)
cr
cr[1]
cr  = np.array_split(br, 2)
cr
cr  = np.array_split(br, 8)
cr

o = np.array(np.vstack([range(10), range(10)]))
print(np.where(o == 3))
o
np.sort([10, 4, 5, 6])
p = np.array(range(10))
np.where(p%2 == 0) # finds the indexes where the values are even = modulo is 0

s = np.array([[1, 2], [1, 2]])
s
r = np.array([[0, 0], [4, 4]])
r, s
r + s
r * s #elemtwise
r**2
r.dot(s)  # "classical" multiplication of two arrays
r.dot?
np.shape(r)
np.shape(r.T)
r.transpose()
r.T
2 * r + 3

ar = np.array(range(12))
ar2 = ar.reshape(2, 6)
ar2
ar3 = ar.reshape(2, 2, 3)
ar3
ar4 = np.array([[1, 2], [2, 3]])
ar5 = ar4.reshape(-1)  #flattening the array
ar5
ar5[ar5 > 2] = 10

np.zeros((3,2))
x = np.eye(9)
x
c = np.diag(x)
c

help(np.diag)
np.diag([7,5,6])
#%%
import numpy as np
from numpy import random

random.randint(100) #generate random integer from  0 to 100 (not including 100 but includig 0)
np.random.rand() #random float between 0 and 1

# random arrays
random.randint(100, size = (2,2))
random.rand(2, 2) #random float between 0 and 1
random.choice([1, 2, 3, 10, 5], size = (10, 10))

random.choice([1, 2, 3], p= [0.3, 0.4, 0.3], size = (3))

# randomly reschuffle and array
y = np.array([1,2,3])
random.shuffle(y) #changes original array
y

random.permutation(y) #same as shuffle, but leaves the original array intact

I = 5000
%time mat2 = np.random.standard_normal((I , I))
%time mat2.sum()
%timeit y.sum()

import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot([0, 1, 2, 3, 3, 3, 6]) #distribution plot
#%% normal distribution
random.normal() #normal distribution, gaussian for 0, 1 
random.normal(0, 1, size = (10, 10))

sns.distplot(random.normal(20, 10, size = 10000))
#%% binomial dist

random.binomial(20, 0.5, 5) #number of succeses in N trials with probability of success p
sns.distplot(random.binomial(n = 10, p = 0.5, size = 1000))

sns.distplot(random.binomial(n = 100, p = 0.5, size = 1000))
sns.distplot(random.normal(50, 5, size = 1000))
# for big number of values the distributions are similar

#%% poisoon distribution
#discrete distribution, tells how how manz times something 
# _can happen given that we know it happens certain amout of time = lam

x = random.poisson(lam = 4, size = 100000)
sns.distplot(x)

# again poisson is similar to normal
sns.distplot(random.normal(loc=30, scale=5, size=10000), hist = False, label='normal')
sns.distplot(random.poisson(lam=30, size=10000), hist = False, label='poisson')

#%% Uniform distribution
x = random.uniform(1, 10, size = (1000))
sns.distplot(x, kde = False)

#%% logistic and lognormal
random.logistic(loc = 1, scale =2, size = 1000)
sns.distplot(random.normal(loc = 0, scale =1, size = 1000), hist = False, label = 'Normal')
sns.distplot(random.logistic(loc = 0, scale =1, size = 1000), hist= False, label = 'logistic')
sns.distplot(random.lognormal(mean = 0.0, sigma = 1, size = 1000)) # can not be negative
# ligistic distribution has more area in the tails, atherwise the distributions are similar

#%% exponential dist

sns.distplot(random.exponential(scale = 2, size = (10, 10)))

#%% Vectorization - ufuncs

a = [1, 2, 3, 4]
b = [10, 20, 30, 40]
z = []

## without vectors we can add to vectors together like:
for i, j in zip(a, b):
    print(i)
    print(j)
    
for i, j in zip(a, b):
    z.append(i + j)
print(z)

## vectorization 
a + b # does not work 
np.array(a) + np.array(b) #works better
np.add(a, b) #works the same

np.multiply(a, b)
np.array(a).dot(np.array(b))
np.add(a, b)
np.sum([a, b, a, b, b], axis = 0)
np.sum([a, b, a, b, b], axis = 1)
np.cumsum(a)
np.prod(a) #product
np.cumprod(a)
np.subtract(a, b)
np.diff([10, 15, 21 ])
np.diff([10, 15, 25, 5 ], n=2) #2nd difference
np.divide(a, b)
np.power(a, b)
np.mod(a, b)
np.remainder(a, b) #same as mod
np.divmod(a, b)
np.absolute(np.array(a)* (-1))

# custom ufunc
def myfunction(x, y):
    return x * y + 3 

multiply = np.frompyfunc(myfunction, 2, 1) # params: func, number of inputs, number of outputs

multiply(a, b)

type(multiply)
type(np.concatenate)

# rounding decimals
np.trunc(100.654898)
np.fix(100.654898)
np.round([100.66489754646, 45.654, 1.6], 2)
np.floor(100.9)
np.ceil(100.1)

np.log(10)
np.log2(4)
np.log10(100)
np.arange(1, 10)

d = 6
f = 4
np.lcm(d, f) # finding lowest common multipl
np.gcd(d, f) # greatest common denominator

a.append(6)
a, d
np.union1d(a, d) #union set of two arrays
np.unique([2, 3, 2]) #unique
np.intersect1d([2, 3], [2])
np.setdiff1d(a, d)
np.setxor1d(a, d)  #values that are NOT present in BOTH sets      
        
