# %% MatPlotLib - Misc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.__version__

mpl.rcParams
plt.rcParams
plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['figure.dpi'] = 100
plt.style.available  # note the fast style
plt.style.use('ggplot')
plt.style.use('bmh')
plt.style.use('seaborn')  # seaborn style
mpl.rcParams['font.family'] = 'serif'

y = np.random.standard_normal(20)
x = range(len(y))

plt.plot(x, y)
plt.axis('equal')  # eqaul scaling for axes

plt.plot(x, y)
plt.axis('tight')

plt.plot(x, y)
plt.axis('Off')  # axis off ---

plt.plot(x, y)
plt.grid(True)  # show grid ---

plt.plot(x, y.cumsum(), 'go', lw=10.5)  # set up style and lineweight ---
plt.plot(x, y.cumsum(), 'b*', lw=6.5)
plt.plot(x, y.cumsum(), 'r-.', lw=2)
plt.xlim(-10, 20)  # sets up x axis ---
plt.xlabel('index x label')  # sets up xlabel ---
plt.ylabel('index y label')
plt.title('titulek')  # sets up title ---

y = np.random.standard_normal((20, 3)).cumsum(axis=0)
plt.plot(y[:, 0], 'r-', lw=1.5, label='1rd label')  # define a label ---
plt.plot(y[:, 1], 'g-', lw=1.5, label='2rd label')
plt.plot(y[:, 2], 'b-', lw=1.5, label='3rd label')
plt.legend(loc=0)  # show legend  "0 is best location" ---

y = np.random.standard_normal((1000, 3))
plt.figure(figsize=(10, 8))  # defines the size x, y---
plt.plot(y[:, 0], y[:, 1], 'r-', lw=0.1)

c = np.random.randint(0, 10, len(y))
plt.scatter(y[:, 0], y[:, 1], c=c)  # scatter graph, c defines the colors
plt.colorbar()  # shows colorbar

plt.hist(y[:, 0], bins=200)  # histogram

# %% Matplotlib official website
# %% https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py

# notes from the matplotlib website
'''
two ways to use Matplotlib:
1] Explicitly create figures and axes - "object-oriented (OO) style"
2] use pyplot functions for plotting    
'''
fig, ax = plt.subplots()  # creates a figure that contatins a single axes
ax.plot([1, 2], [4, 5])

# Versus:
plt.plot([1, 2], [4, 5])

'''
for each Axes graphing method, there is a corresponding function in the matplotlib.pyplot module 
that performs that plot on the "current" axes, creating that axes (and its parent figure) if they don't exist yet.:
'''

fig, ax = plt.subplots()
ax.plot([1, 2], [4, 5])
ax.axes.set_xlim(0, 3)  # sets min and max of x axis, the keyword axes does not have to be included
ax.axes.set_ylim(3, 6)  # there is equivalent plt syntax: plt.xlim
ax.set_title('Title of an axes')  # title
ax.set_xlabel('x_label')
ax.set_ylabel('y_label')
plt.savefig('pokus.png')  # saves the figure
plt.savefig('pokus2.pdf')  # pdf
plt.savefig('pokus2.jpg')  # jpg

'''
All of plotting functions expect numpy.array or numpy.ma.masked_array as input. 
Classes that are 'array-like' such as pandas data objects and numpy.matrix may or may not work as intended. 
It is best to convert these to numpy.array objects prior to plotting.

for example instead of:
    pd.DataFrame(np.random.rand(4, 5), columns = list('abcde'))
use:
    pd.DataFrame(np.random.rand(4, 5), columns = list('abcde')).values

In fact, all sequences are converted to numpy arrays internally.
'''

# %% Pyplot tutorial
# https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py

'''
pyplot API is generally less-flexible than the object-oriented API. 
Most of the function calls can also be called as methods from an Axes object
'''

plt.plot([1, 2], [4, 5], 'r*')
plt.ylabel('ylabel')  # as opposed to set_ylabel()
plt.axis([0, 3, 3, 6])

i = np.arange(0, 10, 0.5)

plt.plot(i, i, 'r*', i, i ** 2, 'bo', i, i ** 3, 'g+')  # probably not the best way...but good to know its doable...

# Plotting with keyword strings
data = {'a': np.arange(50), 'c': np.random.randint(0, 50, 50), 'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['s'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', c='c', s='s', data=data)  # c defines the color, s defines the size

# Plotting with categorical variables
cat = ['cat1', 'cat2', 'cat3']
val = [11, 25, 30]

plt.figure(figsize=(10, 8))
plt.subplot(1, 3, 1)  # nrows, ncols, index. Can be also written as 131 - plt.subplot(131)
plt.bar(cat, val)
plt.subplot(1, 3, 2)
plt.scatter(cat, val)
plt.subplot(1, 3, 3)
plt.plot(cat, val)
plt.suptitle('categories')  # title

# Working with text

mu = 100
x = mu + 15 * np.random.randn(10000)
plt.hist(x, bins=50, alpha=0.5, color='g')  # alpha sets the visibility
plt.text(60, 500, '!Hola', fontsize=20)
plt.grid(True)
plt.xlabel('data', color='red', fontsize=14)  # sets fontsize and color of the label

# matplotlib accepts TeX equation expressions in any text expression, just surround that expresion with $
plt.hist(x, bins=50, alpha=0.5, color='g')
plt.text(60, 500, r'$ x\ =\ \frac{\sqrt{144}}{2}\ \times\ (y\ +\ 12) $', fontsize=20)
# r preceding the title string is important -- it signifies that the string is a raw string and not to treat backslashes as python escapes.

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2 * np.pi * t)
plt.plot(t, s)
plt.annotate('localmax', xy=(2, 1), xytext=(3, 1.5), arrowprops={'color': 'black', 'shrink': 0.5})
plt.xscale('log')  # logarthmic xscale
plt.yscale('linear')  # linear yscale

# %% Sample plots
# https://matplotlib.org/tutorials/introductory/sample_plots.html#sphx-glr-tutorials-introductory-sample-plots-py

## Line Plot

y = np.random.randn(100)
x = np.arange(0, 100, 1)

# playing with various parameters, only one at a time
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=1, color='green', marker='o', markersize=10, linestyle='dashed', alpha=0.4)

plt.plot(x, y, antialiased=False)
plt.plot(x, y, c='m')  # color = c
plt.plot(x, y, dash_capstyle='butt', linestyle='dashed')
plt.plot(x, y, dashes=(5, 2, 1, 2),
         linestyle='dashed')  # describes a sequence of 5 point and 1 point dashes separated by 2 point spaces.
plt.plot(x, y, linestyle='--')
plt.plot(x, y, marker='x', markeredgecolor='red', markersize=10, markevery=10)
plt.grid('True', color=(0.8, 0.6, 0.1), linewidth=1)  # RGB format

# plotting labeled data
import pandas as pd

d = pd.DataFrame(index=[1, 2, 3, 4, 5], data=[10, 20, 35, 40, 50], columns=['Nums'])
plt.plot([0, 1, 2, 3, 4], 'Nums', data=d)

fig, ax = plt.subplots()
ax.plot(x, y, '*-g')  # fmt = '[marker][line][color]'

## Multiple Subplots
fig, ax = plt.subplots(2, 1, frameon=False)
ax[0].plot(x, y)

fig, ax = plt.subplots(2, 1, facecolor='red', sharey='col')
ax[0].plot(x, y)

fig, ax = plt.subplots(2, 1, subplot_kw=dict(polar=True))  # polar axes
ax[0].plot(x, y)

## Images
# barcode
x = np.random.rand(500) > 0.7

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.1, 0.8])
ax.set_axis_off()
ax.imshow(x.reshape((-1, 1)), aspect='auto', cmap='binary', interpolation='nearest')

'''... to be continued with contouring and pseudocolor....'''

# %% Lifecycle of a plot
# https://matplotlib.org/tutorials/introductory/lifecycle.html#sphx-glr-tutorials-introductory-lifecycle-py

import os

os.chdir('F:\_Python')

data = {
    'Barton LLC': 109438.50,
    'Frami, Hills and Schmidt': 103569.59
}

group_data = list(data.values())
names = list(data.keys())
mean = np.mean(group_data)

fig, ax = plt.subplots()
ax.bar(names, group_data)

fig, ax = plt.subplots()
ax.barh(names, group_data)  # horizontal bar

fig, ax = plt.subplots(figsize=(10, 5))  # width high as opposed to conventions in linear algebra
ax.barh(names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45,
         horizontalalignment='right')  # this is the way to set up more properties to one or a list of objects
ax.set(title='Company revenue', xlim=[-10000, 140000], xlabel='Total Revenue',
       ylabel='company')  # sets multiple properties at once
ax.axvline(mean, c='r', ls='--')
for group in [0, 1]:
    ax.text(120000, group, "New Company", fontsize=10, verticalalignment="center")
ax.title.set(y=1.1)
fig.savefig('comps.jpg', transparent=True)

fig.canvas.get_supported_filetypes()  # possibilities to save the file

# %% Numpy


## --- BOX PLOTS
np.random.seed()
x = np.random.randn(1000)
y = np.random.randn(100)
z = np.random.randn(10)
z

from matplotlib import pyplot as plt

fig, ax = plt.subplots()
ax.boxplot((x, y, z), vert=True, showmeans=True, labels=('x', 'y', 'z'), patch_artist=True)

# HISTOGRAM
hist, bin_edges = np.histogram(x, bins=10)

fig, ax = plt.subplots()
ax.hist(x, bin_edges)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')

# PIE CHART
x, y, z = 70, 30, 50

fig, ax = plt.subplots()
ax.pie((x, y, z), autopct='%1.1f%%')

# BAR CHART

x = np.arange(21)
y = np.random.randint(21, size=21)
err = np.random.randn(21)

fig, ax = plt.subplots()
ax.bar(x, y, yerr=err)  # with error bars
ax.set_xlabel('x')
ax.set_ylabel('y')

## Heat Map
matrix = np.cov(x, y).round(decimals=2)

fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')  # Heat map with imshow()




# %% Plotting
import matplotlib.pyplot as plt

df.cumsum().plot(lw=2.0)

numbers = pd.Series(data=[1, 22, 44, 65, 89, 75, 65, 78], name='Counts')
numbers2 = pd.Series(data=[1, 23, 44, None, 9, 76, 65, 100], name='Counts2')

pig = plt.figure()
numbers.plot()
numbers2.plot()
plt.legend()

# bar chart
fig = plt.figure()
numbers.plot(kind='bar')
numbers2.plot(kind='bar', color='g', alpha=0.5)

# histogram
numbers.hist()

numbers.plot(kind='kde')
numbers.plot(kind='density')

numbers.plot.area()  # two ways to specifiy the type of a graph, see below
numbers.plot(kind='area')
numbers.plot.barh()
numbers.plot.box()
numbers.plot(kind='pie')
