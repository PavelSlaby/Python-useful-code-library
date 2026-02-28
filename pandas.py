#%% TIPS
'''
 - if the values are of the same type (int, string...) speed is maximized
 - Typically, built-in methods will be faster because they are vectorized and often implemented in Cython,
    so there is less overhead. Using .map and .apply should be thought of as a last resort, not the first tool you reach for.
'''

#%% General
import pandas as pd
import numpy as np

pd.__version__

#settings
df = pd.DataFrame(np.random.randn(10, 40))
df
pd.set_option('display.max_columns', None) # max number of columns to display, will show all when set to = None
pd.set_option('display.max_columns', 30)
pd.set_option("display.precision", 3)
pd.set_option('display.width', 1000)  # Width of the display in characters
pd.reset_option("^display") # how to reset all options at once
print(df.to_string(line_width=60)) #To limit the width to a specific number of columns, the .to_string method accepts a line_width parameter: print(df.to_string(line_width=60))

df.describe()

df.memory_usage(deep=True) / 1024 ** 2  #memory usage, multiply to be in MB



#%% Series Object
# creating a series
numbers = pd.Series([1, 2, 3, 4, 5])  # index implicit
numbers
numbers = pd.Series([1, 145, 2, 200], name='counts', index=[1, 2, 2, 4])  # explicit is always better
numbers
pd.Series([2, None])  # instead of none automatically displays NaN - not a number, if it can not read a number in clearly otherwise numerical series
pd.Series({1: 'Prague', 2: 'NY'})  # using dict to create series

# investigating a series
numbers = pd.Series([1, 145, 2, 2,  200], index=[1, 2, 'apple', 2,  4])
numbers
numbers['apple']
numbers[2] # returns a series if it finds more than one values, otherwise it returns just a scalar
numbers[1]
numbers[2] = 1
#the previous is not a good practise!! better to use .iloc and .loc

numbers.iloc[-1]  # last
numbers.iloc[[1, 2]]  # can pass a list
numbers.iloc[0:4:2]  # or slice

numbers.loc[['apple', 1]]
numbers.loc['apple':]
numbers.loc[1]

#filtering
numbers[numbers > 100]
numbers == 2
(numbers == 2) | (numbers == 145)

#assignment and other functions
numbers.iloc[2] = None
numbers
numbers.fillna(0)  # fills in the NaN value with 0
numbers
numbers.dropna()
numbers
numbers.notnull()
numbers.isnull()
numbers.dropna(inplace=True)
~numbers.isnull() # can flip a boolean mask by applying the not operator (~):

numbers.first_valid_index()
numbers.last_valid_index()

numbers.count()  # counts non NaN values
numbers.value_counts()
numbers.unique()  # which are uniques?
numbers.nunique()  # number of uniques
numbers.drop_duplicates()
numbers.sort_values()

del numbers[1]

# in series iteration is over the values, membership is over the index, see below:
200 in numbers  # returns False, it checks against the index
200 in set(numbers)  # works
2 in numbers  # works because it checks against the index which contains 2

# simular to dictionaries, we can iterate only keys:
for i in numbers.keys():
    print(i)

numbers.repeat(2)  # simply repeats each items a number of times

# conversion
numbers.astype(str)
pd.to_numeric(numbers)  # converts to numeric
pd.to_datetime(numbers)

# index operations
numbers.index
numbers.index.is_unique  # index does not have to be unique
numbers.reset_index()  # resets the index to numbers
numbers.reset_index(drop=True)  # drops the existing index column
numbers.reset_index(drop=True, inplace=True)  # Inplace
numbers = numbers.reindex([2, 1, 0, 10])  # re-indexes - if the new index value is not in the original series, NaN will appear
numbers.rename({10: 3})  # only updates the name of an index label
numbers.sort_index()  # sorts by index values

# string operations
names = pd.Series(['zeus', 'Martin', 'Luke', 'James', 'Alzbeta'])
names.str.lower()
names.str.findall('a')
names.str.count('a')
names.str.join('-')
names.str.len()


#%% DataFrame
# DataFrame creation and investigation
pd.DataFrame([[10, 20, 30, 40], [1, 2, 3, 4]])
df = pd.DataFrame([[10, 20, 30, 40], [1, 2, 3, 40]], columns=['col1', 'col2', 'col3', 'col4'], index=['row1', 'row2']) #explicit better then implicit
df

df['5th'] = [3, 2]
df

a = [1, 2, 3, 4, 5]
a[2:6][0] = 50  # does not work, does not assign anything, python creates temp object
a[0] = 50 #this works, because it is not performed on a slice
a[2:6] = 40, 40, 40 #this also works

a = [1, 2, 3, 4, 5]
a[2:6] = 40 # does not work, because 40 is not iterable
a[2:6] = [40] # I can make it a list, but then the the original will get a bit truncated....
a[2:6] = [40, 40, 40] #also works

a = df.set_index('col4')  # sets index based on existing column
a
a.columns
a.index
a.axes
a.axes[1] # 0 is the index, and 1 is the column
a.values
type(a.index)
type(a.columns)
type(a.values)
a.shape
a.info()

df.set_index('col4', verify_integrity=True)  # verifies integrity, if there are duplicates, throws an error
df.insert(0, 'col5', [0, 1])  # insert a column at a specified location

df.replace(40, 80)
df.loc[:, 'col2'].iloc[-2:]  # the last 2rows in a column

df['col1'].name
df['col1'].dtype
df['col1'].index

#creatig DF from a Dictionary
pd.DataFrame({'growth': [1, 2, 3, 4, 5], 'string': ['guten', 'bye', 'sula', 'hopla', 'A']})

# creating dataframe row by row, #it is a list of dictionaries
pd.DataFrame([
    {'growth': 1, 'string': 'guten'},
    {'growth': 2, 'string': 'bye'},
    {'growth': 3, 'string': 'sula'},
    {'growth': 4, 'string': 'hopla'}
])

dfnum = pd.DataFrame(np.random.randn(10, 3), columns=['a', 'b', 3])
dfnum


for i in df:
    print(i)  # iteration goes over the column names

for i in df.keys():
    print(i)  # more explicit

for i in df.iterrows():  # over the rows
    print(i)

# delete columns with .pop, .drop, or del
a = df.drop(['row1', 'row2'])  # drops rows
a
del df['col1']
df
df.pop('col2')
df.drop('col3', axis=1)  # to drop a column, meaning "apply this to a column"

df = pd.DataFrame([[10, 20, None, 40], [None, 2, 3, 40]], columns=['col1', 'col2', 'col3', 'col4'],
                  index=['row1', 'row2'])
df
df.isnull().any()  # if any value in column is True
df.fillna(method='ffill')
df.fillna(method='bfill')
df.fillna(method='bfill', axis=1)
df.interpolate()
df.replace(np.nan, 'ahoj')

#%% SELECTING ITEMS
df = pd.DataFrame(data=np.array([['AK', 'blue', 'Apple', 30, 165, 4.6],
                                 ['US', 'green', 'Pork', 2, 70, 8.3],
                                 ['FL', 'red', 'Mango', 12, 120, 9.0],
                                 ['AL', 'white', 'Apple', 4, 80, 3.3],
                                 ['AK', 'gray', 'Cheese', 32, 180, 1.8],
                                 ['TX', 'black', 'Melon', 33, 172, 9.5],
                                 ['TX', 'red', 'Beans', 69, 150, 2.2]], dtype=object)
                  , columns=pd.Index(['state', 'color', 'food', 'age', 'height', 'score'], dtype=object)
                  # object can be specified with or w/o quotation marks
                  , index=pd.Index(['Jane', 'Niko', 'Aaron', 'Penelope', 'Dean', 'Christina', 'Cornelia'],
                                   dtype='object'))

df

df[df.height > 150]
df.query('age > 10')  # alternative to previous, but does not work well with numerical columns
df.reindex(index=['Jane'], columns=['color', 'height'])  # selects based on specified index

## SELECTING WITH .LOC or COLUMN LABEL
df['food']  # works only for columns
df.food  # not practical....
df[['food', 'color']]  # list of columns, not recommended!!!!!!!!
df[['food', 'food']]  # selecting the same column twice
df['Jane': 'Cornelia'] # and thats why these are confusing
df['state': 'food']  # does not work

df[3]  # does not work, but a slice would work
df[3:5]  # slices work for rows - but are impractical
df['Aaron': 'Christina']  # also works for rows but not recommended for use

type(df['food'])  # pandas.core.series.Series
df['food'].index #remembers the index even if selecting just one column
df['food'].values
df['food'].name  # this is the old column name, but series has no columns

# Series has two main components, the index and the data, NO columns in a Series!
df[['color', 'score', 'food']]  # list of multiple or just one column
df[['color']]

# .loc selects by either a columns or row LABEL
df.loc['Niko'] # first parameter is the row label, column label can be left out....
df.loc[['Niko', 'Penelope']]
df.loc['Niko': 'Penelope']  # includes the last value (as opposed to other python lists etc)
df.loc[:'Penelope']  # other typical slices are possible
df.loc['Penelope':]
df.loc[:'Penelope':2]

df.loc['food']  # does not work, with .loc, the row parameter is mandatory! - but the column parameter can be left out
df.loc[:, 'food']
df.loc['Niko':'Christina':3, 'color':'score':2]  # various combinations are possible
df.loc[['Niko', 'Christina'], 'color':'score':2]

## SELECTING WITH .ILOC (integer location)
# works the same as loc but uses integer location instead of labels
df.iloc[:]
df.iloc[[3, 4]]
df.iloc[1:]
df.iloc[:-1:2]
df.iloc[::2]  # all but every second
df.iloc[1:6, [4, 5]]

# difference between using .iloc and .loc is that iloc excludes the last value, and loc includes it
df.iloc[1:3]
df.loc['Niko': 'Penelope']

## Series
# does not have columns, so it only selects based on index (row of a dataframe), only one indexer is thus needed
# series has only index and the values
food = df['food']
food.index

food.Jane  # selecting by the index - does not work on a DataFrame - there it works only on columns
food['Jane']  # works but better is to use .loc or .iloc

food.loc['Jane']
food.loc['Jane'::2]  # various selections are possible
food.loc[['Jane', 'Aaron']]

food.iloc[1]  # returns a string
food.iloc[1::2]  # returns a series
food.iloc[[1, 5, 2]]

type(food.iloc[1])
type(food.iloc[1::2])

food.tolist()[1:2]
food[1:2]

# in general its better to use .loc or .iloc
food[2:4]  # also not recommended
food['Aaron':'Penelope']  # also not recommended
food[['Aaron', 'Penelope']]  # also not recommended

new_series = pd.Series(['Praha', 'Brno', 'Olomouc', 'Dresden', 'Moenchengladbach', 'Berlin', 'Muenster'])
df['Series'] = new_series  # Causes NaN!! because the indexes are different! indexes have to match!
df
df.index == new_series.index #throws false

df.index = new_series.index #if I change the index, it will work
df['Series'] = new_series
df

df.score = df.score.astype(int)  # changing type of column


#%% Subsets
from pathlib import Path
directory = Path(r'D:\_Python\General useful')
file = Path(r'QueryResults.csv')

f = pd.read_csv(directory / file)
ten = f.head(10)
f.info()

ten[[True, False, False, 'a' == 'n', True, True, False, 1 > -1, 1 == 1,
     False]]  # list of booleans, points to rows to select

criterion = f['score'] == 5  # assigns booleans series to criterion
f[criterion]
type(f['score'] >= 5)  # is pandas series with booleans

# .loc can be used in the exact same way, or column parameters can be added
f.loc[criterion, 'score'::3]

'''
Only the following operators work with pandas:
& - and (ampersand)
| - or (pipe)
~ - not (tilde)
'''

f[(f.score > 4) and (f.commentcount > 2)]  # does not work (as expected)
f[(f.score > 4) & (f.commentcount > 2)]
f[f.score > 4 & f.commentcount > 2]  # does not work

f.loc[(f.score > 4) & (f.commentcount > 2), 'quest_name'] #using the loc. is the best approach - most readable
f[(f.score > 4) & (f.commentcount > 2)]['quest_name'] #but this also works...

f.loc[(f.score > 4) | (f.commentcount > 2), :]
f.loc[(f.score > 4) & ~ (f.commentcount > 2), :]  # and not
f.loc[~((f.score > 4) & ~ (f.commentcount > 2)), :]  # whole condition wrapped and tilde used

# when there are a lot of OR conditions, two options:
# 1]
criterion = ((f.quest_name == 'Atnhai Atthawit') |
             (f.quest_name == 'burcak')
             )

f.loc[criterion, :]

# 2]
f.loc[f['quest_name'].isin(['Atnhai Atthawit', 'burcak'])]  # ISIN

f.loc[f['quest_name'].isnull()]  # find a column with NaN
f.loc[f['quest_name'].isna()]  # alias of isnull
f.loc[f['commentcount'].between(5, 7)]  # between method

# booleans for columns
f.loc[:, ['answercount', 'commentcount']].loc[:, f[['answercount', 'commentcount']].mean() > 1]



df[df.state == 'US'][
    'color'] = 'red'  # SettingWithCopyWarning  thats why I should use .loc  https://www.dataquest.io/blog/settingwithcopywarning/

df.loc[df.state == 'US', 'color'] = 'red'

df.state.Jane = 'MN'

df.sort_index()
df = df.set_index('color')
df.sort_index()['bl':'red']  # when the index is sorted, I can use indexing by letter...weird?!


#%% Connecting dataframes together
# Concatenate
df1 = pd.DataFrame({'name': ['John', 'George', 'Ringo'], 'color': ['Blue', 'Blue', 'Purple']})
df2 = pd.DataFrame({'name': ['Paul', 'George', 'Ringo'], 'carcolor': ['Red', 'Blue', np.nan]}, index=[3, 1, 2])
df1
df2

pd.concat([df1, df2])  #by default it concatenates by axis = 0
pd.concat([df1, df2], axis=1) #if I specify axis = 1 then it works more like a join
pd.concat([df1, df2], verify_integrity=True)  # throws an error as expected
pd.concat([df1, df2], ignore_index=True)  # creates a new index

#JOIN
# join is done joining based on indexes, not columns
df2.join(df1)  #if column names have the same name, you need to specify suffix as below, otherwise will throw an error
df1.join(df2, lsuffix= "_left", rsuffix= "_right")  # by default it is a "left join"
df1.join(df2, how='right',  lsuffix= "_left", rsuffix= "_right")
df1.join(df2, how='inner',  lsuffix= "_left", rsuffix= "_right")
df1.join(df2, how='outer',  lsuffix= "_left", rsuffix= "_right")

df1.set_index('name').join(df2.set_index('name'))  # work like left join

# MERGE method is preferable:
df1.merge(df2)  # defualt mode is inner join
df1.merge(df2, how='outer')
df1.merge(df2, how='left')
df1.merge(df2, how='right')

df1 = pd.DataFrame(data=[100, 200, 500, 400], columns=['A'], index=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(data=[200, 50, 150], columns=['B'], index=['f', 'b', 'd'])
df1
df2

c = pd.Series([250, 150, 50], index=['b', 'd', 'c'])
df1['C'] = c
df2['C'] = c  # !!! kinda like "left merge"

pd.merge(df1, df2)  # mergin on column C
pd.merge(df2, df1)  # mergin on column C
pd.merge(df2, df1, on='C')  # mergin on column C
pd.merge(df2, df1, how='outer')
pd.merge(df1, df2, left_on='A', right_on='B', how='outer')  # joins on A from left table, on B fro right
pd.merge(df1, df2, left_on='A', right_on='B', how='inner')
pd.merge(df1, df2, left_index=True, right_index=True)
pd.merge(df1, df2, left_index=True, right_index=True)

# Grouping
df['Quarter'] = ['Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3']
df

groups = df.groupby('Quarter')
groups
groups.max()
groups.size()
groups.mean()
groups.aggregate(['min', 'max', 'size', 'count']).round(2)

df['Odd_Even'] = ['Odd', 'Even', 'Odd', 'Even', 'Odd', 'Even', 'Odd']
df
group = df.groupby(['Quarter', 'Odd_Even'])
group.size()
group.mean()

# pivot table
scores = pd.DataFrame({
    'name': ['Adam', 'Bob', 'Dave', 'Fred'],
    'age': [15, 16, 16, 15],
    'test1': [95, 81, 89, None],
    'test2': [80, 82, 84, 88],
    'teacher': ['Ashby', 'Ashby', 'Jones', 'Jones']})

scores.pivot_table(index='teacher')
scores.pivot_table(index='teacher', values=['test1', 'test2'], aggfunc='median')
scores.pivot_table(index=['teacher', 'age'], values=['test1', 'test2'], aggfunc=['median', 'max'])

# melting
pd.melt(scores, id_vars=['name', 'age'], value_vars=['test1', 'test2'])
pd.melt(scores, id_vars=['name'], value_vars=['test1', 'test2', 'teacher', 'age'], var_name='The Parameter')

# dummy variable
pd.get_dummies(scores, columns=['age'])

#%% data_range
dates = pd.date_range('2015-01-01,', periods=9, freq='ME')
dates1 = pd.date_range(start='2015-01-01', end='2016-01-01', freq='ME')
dates
dates1

pd.date_range(start='2015-01-01', end='2016-01-01', freq='ME')  # month end
pd.date_range(start='2015-01-01', end='2016-01-01', freq='MS')  # months start
pd.date_range(start='2019-01-01', end='2020-01-01', freq='B')  # biz day freq
pd.date_range(start='2019-01-01', end='2020-01-01', freq='D')  # calendar day freq
pd.date_range(start='2019-01-01', end='2020-01-01', freq='BME')  # biz month end day freq
pd.date_range(start='2019-01-01', end='2020-01-01', freq='QE')
pd.date_range(start='2010-01-01', end='2020-01-01', freq='min')  # minutely

start, end = '2021-01-01', '2021-02-02'
dates = pd.DataFrame(index=pd.date_range(start, end))  # creating empty dataframe, only the index is defined

# convert datatypes
pd.to_datetime(dates.index, format="%Y-%m-%d %H:%M:%S").strftime("%Y")
pd.to_numeric(dates.index)
scores.test1.apply(pd.to_numeric)
dates.index.strftime('%m / %Y')

#%% Math Operations
a = np.random.standard_normal((9, 4))
a = pd.DataFrame(a)
a.round(2)

a // 2  # "floor" divides series
a.rank()
a.rank(ascending=False)
df.describe()
df.transpose()
df.T

df.sum()
df.prod()  # product
df.sum(axis=0)  # axis are the same as for numpy
df.sum(axis=1)
df['age'].quantile(.01)
df['age'].skew()
df['age'].kurt()
df['age'].diff()  # first difference of a series
df['age'].clip(lower=3, upper=8)  # clipped values, changes the min max to the parameters

df.loc[:, 'age'].apply(lambda x: x ** 2)

df.loc[:, 'age'] ** 2
df.loc[:, 'age'] + 2
df.loc[:, 'age'] * 1.0 - 1

numbers.dot(numbers)

df.loc[:, 'age'].mean()
df[['age', 'height']].mean()

np.array(df)  # to generate ndarray from dataframe

a.corr()
a.cov()
a.count()
a.mean()
a.std()
a.cumsum()
a.abs()
a.add(20)
a.nlargest(3, 2)  # Return the first `n` rows ordered by `columns` in descending order.
df = np.sqrt(df)
df.sum()

s = pd.Series(np.random.randint(0, 10, 100))
s

s.head(9)
s += 100
s

x = [8.0, 1, 2.5, 4, 28.0]
y = [9.0, 10, 1.5, 40, 28.0]

v = pd.Series(x)
u = pd.Series(y)
v.corr(u)

import math
pd.Series([1.0, math.nan, 4]).mean()  # pandas ignores nan by default X numpy

# %% Getting data from yfinance
import yfinance as yf

df = yf.download('AAPL', start='2000-01-01', end='2010-12-31',
                       actions='inline')  # actions: includes stock spkits + dividends

df['simple_rtn'] = df.loc[:, 'Close'].pct_change()
df['log_rtn'] = np.log(df.loc[:, 'Close'] / df.loc[:, 'Close'].shift(1))

index1 = pd.date_range(start='1999-12-31', end='2010-12-31')
dates = pd.DataFrame(index=index1)

dates.join(df.Close, how='left', ).fillna(method='ffill')

df = df.asfreq('ME')  # najde posledni den v mesici a vezme jeho hodnotu, pokud zadna hodnota neni (napr vikend) doplni NaN, proto lepsi pouzit nasledujici:
df = df.resample('ME').last()
# alternatively:
df["log_rtn"].resample('ME').mean()  # average monthly return

def realized_vol(x):
    return np.sqrt(np.sum(x ** 2))





