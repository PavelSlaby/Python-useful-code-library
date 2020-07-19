# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:44:12 2020

@author: pavel_000
"""

## MatPlotLib

a = 5
del a

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

%matplotlib inline

np.random.seed(1000)

y = np.random.standard_normal(20)
type(y)
y
np.shape(y)

x = range(len(y))
x

mpl.pyplot.plot(x , y)
plt.plot(x, y)
plt.plot(x)

t = np.random.standard_t(10, 20)

plt.plot(x, t.cumsum())
plt.grid(1)
plt.axis('Off')

plt.plot(x, y.cumsum(), 'go', lw = 10.5)
plt.plot(x, y.cumsum(), 'b*', lw = 6.5)
plt.plot(x, y.cumsum(), 'r-.', lw = 2)
plt.xlim(-10, 20)
plt.xlabel('index x label')
plt.ylabel('index y label')
plt.title('titulek')

y = np.random.standard_normal((20,3)).cumsum(axis=0)
np.shape(y)
plt.plot(y[:, 0],  'r-' , lw = 1.5, label = '1rd label')
plt.plot(y[:, 1],  'g-' , lw = 1.5, label = '2rd label')
plt.plot(y[:, 2],  'b-' , lw = 1.5, label = '3rd label')
plt.legend(loc = 0)
plt.grid(1)
plt.xlabel('xpopisek')
plt.ylabel('ypopisek')
plt.title('Titulek')
plt.axis('tight')


y = np.random.standard_normal((20,3)).cumsum(axis=0)
y[:, 1] = y[:, 1] * 100
plt.plot(y[:, 0],  'r-' , lw = 1.5, label = '1rd label')
#plt.plot(y[:, 1],  'g-' , lw = 1.5, label = '2rd label')
#plt.plot(y[:, 2],  'b-' , lw = 1.5, label = '3rd label')
plt.legend(loc = 0)
plt.grid(1)
plt.xlabel('xpopisek')
plt.ylabel('ypopisek')
plt.title('Titulek')
plt.axis('tight')

y = np.random.standard_normal((1000,3))
plt.figure(figsize = (10,10))
plt.plot(y[:, 0], y[:, 1], 'ro', lw = 0.1)    
plt.grid(1)
plt.title('scatter')

y = np.random.standard_normal((1000,3))
plt.scatter(y[:, 0], y[:, 1])    

c = np.random.randint(0, 10, len(y))
plt.scatter(y[:, 0], y[:, 1], c=c)
plt.colorbar()

y = np.random.standard_normal((1000, 3))
plt.hist(y[:, 0], bins= 200)

strike = np.linspace(50, 150, 24)
ttm = np.linspace(0.5, 2.5, 24)

strike, ttm = np.meshgrid(strike, ttm)
strike, ttm

iv = (strike - 100) **2 / (100 * strike) /ttm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize = (9,6))
ax = fig.gca(projection = '3d')
surf = ax.plot_surface(strike, ttm, iv, cmap=plt.cm.coolwarm)
ax.set_xlabel('strike')
ax.set_ylabel('time-to-maturity')
ax.set_zlabel('implied volatility')
fig.colorbar(surf)


