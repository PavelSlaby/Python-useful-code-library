# %% MatPlotLib overview
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#%% General
mpl.__version__
mpl.rcParams
plt.rcParams
plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['figure.dpi'] = 100
plt.style.available  # note the fast style
plt.style.use('ggplot')
plt.style.use('bmh')
mpl.rcParams['font.family'] = 'serif'




# %% Matplotlib official - Quick Start Guide
# %% https://matplotlib.org/stable/users/explain/quick_start.html

# notes from the matplotlib website
'''
All of plotting functions expect numpy.array or numpy.ma.masked_array as input. It is best to convert Pandas and others  to numpy.array objects prior to plotting.

    for example instead of:     pd.DataFrame(np.random.rand(4, 5), columns = list('abcde'))
    use:                        pd.DataFrame(np.random.rand(4, 5), columns = list('abcde')).values
'''


'''
two ways to use Matplotlib:
1] Explicitly create figures and axes - "object-oriented (OO) style"
2] use pyplot functions for plotting    
'''
#1] OOP:
fig, ax = plt.subplots()  # creates a figure that contains a single axes (axes can be loosely understood as a graph..., figure is more like canvas on which you can put multiplce graphs)
ax.plot([1, 2], [4, 5])
plt.show() #its best practice to include this, but depending on the IDE, it mig be left out...

#2] Versus:
plt.plot([1, 2], [4, 5])

'''
for each Axes graphing method, there is a corresponding function in the matplotlib.pyplot module 
that performs that plot on the "current" axes, creating that axes (and its parent figure) if they don't exist yet.:
'''


#typical graph: OOP
fig, ax = plt.subplots(figsize = (10, 10))
ax.plot([1, 2], [4, 5], label = '1st label')
ax.plot([0, 3], [3, 3.5], label = '2nd label')
ax.axes.set_xlim(0, 3)  # sets min and max of x axis, the keyword axes does not have to be included
ax.axes.set_ylim(3, 6)  # there is equivalent plt syntax: plt.xlim
ax.set_title('Title of an axes')  # title
ax.set_xlabel('x_label')
ax.set_ylabel('y_label')
ax.legend()

#the same in pyplot style
plt.plot([1, 2], [4, 5], label = '1st label')
plt.plot([0, 3], [3, 3.5], label = '2nd label')
plt.xlabel('x_label')
plt.ylabel('y_label')
plt.ylim(3, 6)
plt.xlim(0, 3)
plt.title('Title of an axes')
plt.legend()

'''
pyplot API is generally less-flexible than the object-oriented API. 
Most of the function calls can also be called as methods from an Axes object
'''

# multiple axes
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2) # specifiec how many axes to create...
ax4.plot([1, 2], [4, 5])
ax3.plot([1, 2], [1, 1])
ax1.plot([1, 2], [1, 1])
ax2.plot([1, 2], [1, 1])

# or
fig, axes = plt.subplots(2, 2) # specifiec how many axes to create...
axes[0, 0].plot([1, 2], [4, 5])
axes[0, 1].plot([1, 2], [1, 1])
axes[1, 0].plot([1, 2], [1, 1])
axes[1, 1].plot([1, 2], [1, 1])

#or
cat = ['cat1', 'cat2', 'cat3']
val = [11, 25, 30]
plt.subplot(1, 3, 1)  # nrows, ncols, index. Can be also written as 131 - plt.subplot(131)
plt.bar(cat, val)
plt.subplot(1, 3, 2)
plt.scatter(cat, val)
plt.subplot(1, 3, 3)
plt.plot(cat, val)





# more ways to style the graphs
plt.savefig('pokus.png')  # saves the figure

i = np.arange(0, 10, 0.5)
plt.plot(i, i, 'r*', i, i ** 2, 'bo', i, i ** 3, 'g+')  # probably not the best way...but good to know its doable...

y = np.random.standard_normal((20, 2))
x = range(len(y[:,0]))
fig,ax = plt.subplots(layout = 'constrained')
ax.plot(x, y[:,0].cumsum(), 'go', lw=10.5)  # shortcuts for setting up style and color
ax.plot(x,   y[:,1].cumsum(), 'b*', lw=6.5)
ax.axis('off')


fig, ax = plt.subplots()
ax.plot(x, y[:,0].cumsum(), color = 'red', linestyle = '--', linewidth=10.5, alpha = 0.3, label = '1st label')
a, = ax.plot(x,   y[:,1].cumsum(), color = 'orange', linestyle = '--',  linewidth=6.5) # ax.plot() returns a list of line2D objects...so with the comma, it unpacks a single element
a.set_linestyle(':') # styling can be also set after later....
ax.set_xlabel('x', color = 'red')
ax.text(0, 0, 'text field' , fontsize = 14)
ax.set_xticks([0,1,2,3,4,5])
ax.grid(False)
ax.legend(loc= 0)  # show legend  "0 is best location"
ax.text(10, -2, r'$ x\ =\ \frac{\sqrt{144}}{2}\ \times\ (y\ +\ 12) $', fontsize=20)
#TeX equation expressions in any text expression, just surround that expresion with $
# r preceding the title string is important -- it signifies that the string is a raw string and not to treat backslashes as python escapes.

#another example
y = np.random.randn(100)
x = np.arange(0, 100, 1)

# playing with various parameters, only one at a time
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=1, color='green', marker='o', markersize=10, linestyle='dashed', alpha=0.4)
ax.plot(x, y, antialiased=False)
ax.plot(x, y, c='m')  # color = c
ax.plot(x, y, dash_capstyle='butt', linestyle='dashed')
ax.plot(x, y, dashes=(5, 2, 1, 2), linestyle='dashed')  # describes a sequence of 5 point and 1 point dashes separated by 2 point spaces.
ax.plot(x, y, linestyle='--')
ax.plot(x, y, marker='x', markeredgecolor='red', markersize=10, markevery=10)
ax.grid('True', color=(0.8, 0.6, 0.1), linewidth=1)  # RGB format


#others:
plt.plot( y[:, 0],  'r-', lw=1.5, label='1rd label')  # here I dont have the x defined explicitly, it implictly assumes 1 spacing...


#%%TYPEs OF Graphs --------------------------------------------------------

#SCATTER PLOT
data = {'a': np.arange(50), 'c': np.random.randint(0, 50, 50), 'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['s'] = np.abs(data['d']) * 100

fig, ax = plt.subplots()
a = ax.scatter('a', 'b', c='c', s='s', data=data, edgecolor = 'k')  # c defines the color, s defines the size
plt.colorbar(a)  # shows colorbar


# HISTOGRAM
mu, sigma = 0, 1
x = mu + sigma * np.random.randn(10000)
hist, bin_edges = np.histogram(x, bins=100)

fig, ax = plt.subplots()
ax.hist(x,  bin_edges)


# PIE CHART
x, y, z = 70, 30, 50

fig, ax = plt.subplots()
ax.pie((x, y, z), autopct='%1.1f%%')



## --- BOX PLOTS
x = np.random.randn(1000)
y = np.random.randn(100)
z = np.random.randn(10)

fig, ax = plt.subplots()
ax.boxplot((x, y, z), vert=True, showmeans=True, labels=('x', 'y', 'z'), patch_artist=True)




## Images
# barcode
x = np.random.rand(500) > 0.7

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.1, 0.8])
ax.set_axis_off()
ax.imshow(x.reshape((-1, 1)), aspect='auto', cmap='binary', interpolation='nearest')

'''... to be continued with contouring and pseudocolor....'''



data = {
    'Barton LLC': 109438.50,
    'Frami, Hills and Schmidt': 103569.59
}

group_data = list(data.values())
names = list(data.keys())
mean = np.mean(group_data)


fig, ax = plt.subplots(figsize=(10, 5))  # width high as opposed to conventions in linear algebra
ax.barh(names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')  # this is the way to set up more properties to one or a list of objects
ax.set(title='Company revenue', xlim=[-10000, 140000], xlabel='Total Revenue',       ylabel='company')  # sets multiple properties at once
ax.axvline(mean, c='r', ls='--')
for group in [0, 1]:
    ax.text(120000, group, "New Company", fontsize=10, verticalalignment="center")
ax.title.set(y=1.1)






# BAR CHART

x = np.arange(21)
y = np.random.randint(21, size=21)
err = abs(np.random.randn(21))

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




# %% Plotting with pandas
import matplotlib.pyplot as plt


numbers = pd.Series(data=[1, 22, 44, 65, 89, 75, 65, 78], name='Counts')
numbers2 = pd.Series(data=[1, 23, 44, None, 9, 76, 65, 100], name='Counts2')

numbers.plot()
numbers2.plot()

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
