#%% -- General
import numpy as np
import pandas as pd

pd.__version__

pd.set_option('display.max_columns', None)
pd.set_option('display.max_columns', 500)
pd.set_option("precision", 1)
pd.set_option('display.width', 100) #Width of the display in characters

'''
To limit the width to a specific number of columns, the .to_string method accepts a line_width parameter:
print(df.to_string(line_width=60))
'''

#if the values are of the same type (int, string...) speed is maximized

"""
Typically, built-in methods will be faster because they are vectorized and often implemented in Cython, 
so there is less overhead. Using .map and .apply should be thought of as a last resort, 
instead of the first tool you reach for.
"""

#%% Series structure

#creating a series
pd.Series([1, 145, 2, 200], name  = 'counts') #index implicit
numbers = pd.Series([1, 145, 2, 200], name  = 'counts', index  = [1, 2, 2, 4]) #explicit is always better
pd.Series([2, None]) # istead of none automatically displays NaN - not a number, if it can not read a number in clearly otherwise numerical series
pd.Series({1: 'Prague', 2: 'NY'}) #using dict to create series

#investigating a series
numbers = pd.Series([1, 145, 2, 200], index  = [1, 2, 'apple', 4])
numbers['apple']
numbers[2]
numbers[2] = 1

# the previous is not a good practise!! better to use .iloc and .loc
numbers.iloc[-1] #last
numbers.iloc[[1, 2]] # can pass a list
numbers.iloc[0:4:2] # or slice

numbers.loc[['apple', 1]]
numbers.loc['apple' :]

numbers[numbers > 100]
numbers[2] #returns a series if it finds more than one values, otherwise it returns just a scalar

numbers == 2
(numbers == 2) | (numbers == 145) 

numbers.iloc[2] = None
numbers
numbers.fillna(0) #fills in the NaN value with 0
numbers
numbers.dropna()
numbers
numbers.notnull()
numbers.isnull()
numbers.dropna(inplace = True)
#can flip a boolean mask by applying the not operator (~):
~numbers.isnull()

numbers.first_valid_index()
numbers.last_valid_index()

numbers.count() #counts non NaN values
numbers.value_counts()
numbers.unique() #which are uniques?
numbers.nunique() #number of uniques

numbers.drop_duplicates()
numbers.sort_values()

#iteration
# in series iteration is over the values, membership is over the index, see below:
145 in numbers #returns False, it checks against the index
145 in set(numbers) #works
2 in numbers #works because it checks against the index which contains 2

# to iterate over the values, use iteritems()
for i in numbers.iteritems():
    print(i)

#simular to dictionaries, we can iterate only keys:
for i in numbers.keys():
    print(i)
    

numbers.repeat(2) #simply repeats each items a number of times
    
#index operations
numbers.index
numbers.index.is_unique #index does not have to be unique

numbers.reset_index() #resets the index to numbers
numbers.reset_index(drop = True) # drops the actul index column

numbers.reset_index(drop = True, inplace = True) # Inplace
numbers = numbers.reindex([2,1, 0, 10]) #reindexes - if the new index value is not in the original series, NaN will appear
numbers.rename({10: 3}) #only updates the name of an index label
numbers.sort_index() #sorts by index values

#string operations
names = pd.Series(['zeus', 'Martin', 'Luke', 'James', 'Alzbeta'])
names.str.lower()
names.str.findall('a')
names.str.count('a')
names.str.join('-')
names.str.len()

#conversion
numbers.astype(str)
pd.to_numeric(numbers) #converts to numeric
pd.to_datetime(numbers)

#other    
del numbers[2]    


#%%File operations
numbers = pd.Series([1, 145, 2, 200], name  = 'counts')

file = "f:\_Python\General useful\/python_test.csv"

with open(file,"w") as f:
    numbers.to_csv(file, header = True, index_label = 'Index')

open(file,"r")
pd.read_csv(file, index_col = 0)

#%% DataFrame

#DataFrame creation and investigation
pd.DataFrame([[10, 20, 30, 40],[1, 2, 3, 4]])
df = pd.DataFrame([[10, 20, 30, 40],[1, 2, 3, 40]], columns = ['col1', 'col2', 'col3', 'col4'], index = ['row1', 'row2'])
df

df.reindex(index = ['row1'], columns = ['col2', 'col4']) #selects based on specified index
df.set_index('col4') #sets index based on existing column
df.set_index('col4', verify_integrity = True) #verifies integrity, if there are duplicates, throws an error
df.insert(0, 'col5', [0, 1]) #insert a column at a specified location

df.replace(40, 80)
df.loc[:, 'col2'].iloc[-2:] #the last 2rows in a column

df.axes # 0 is the index, and 1 is the column 
df.axes[1]
df.index
df.columns
df.values
df.shape
df.info()


df['col1'].name
df['col1'].dtype
df['col1'].index

pd.DataFrame({'growth' : [1, 2, 3, 4, 5], 'string' : ['guten', 'bye', 'sula', 'hopla', 'A']})

#creating dataframe row by row, #it is a list of dictionaries
pd.DataFrame([ 
    {'growth' : 1, 'string': 'guten'},
    {'growth' : 2, 'string': 'bye'},
    {'growth' : 3, 'string': 'sula'},
    {'growth' : 4, 'string': 'hopla'}
            ])

dfnum = pd.DataFrame(np.random.randn(10, 3), columns = ['a', 'b', 3])

## Basic manipulation
df['5th'] = [3, 2]
df

df = df.append({'numbers': 3, '2': 3.1, '3': 5.5, '4': 4.4, '5th': 3.5 ,'6th': 3.4 }, ignore_index = 1) #replaces the index!!! and creates a new row if it does not know to which of existing rows to assging the new data
df1 = df.append({'numbers': 3, '2': 3.1, '3': 5.5, '4': 4.4, '5th': 3.5 ,'6th': 3.4 }) #does not know to which row to assign
df1 = df.append(pd.DataFrame({'col1': 3, 'col2': 3.1, 'col3': 5.5, 'col4': 4.4,}, index = ['row3'])) #remedies the problem above
df2 = df.append(pd.DataFrame({'col1': 3, 'col2': 3.1}, index = ['row3'])) #incomplete row results in NaN
df2.values #np array object 

for i in df:
    print(i) #iteration goes over the column names
    
for i in df.keys():
    print(i) #more explicit
    
for i in df.iteritems():
    print(i) 
    
for i in df.iterrows(): #over the rows
    print(i) 
    

#delete columns with .pop, .drop, or del
df.drop([1,2]) # drops rows
del df['col1']
df.pop('2')
df.drop('3', axis = 1) # to drop a column, meaning "apply this to a column"

pd.concat([df, df])
pd.concat([df, df], ignore_index = True) #ignores the index
df.append(df) #same as concat?

a  = np.random.standard_normal((9, 4))
df2 = pd.DataFrame(a, index = range(11, 20), columns = ('a', 'b', 'c', 'd'))
df2

df2.columns = ['1st' ,'2nd', '3rd', '4th']
df2['2nd'].iloc[2]

#data_range
dates = pd.date_range('2015-01-01,', periods = 9, freq = 'M')
dates1 = pd.date_range(start = '2015-01-01', end = '2016-01-01', freq = 'M')
dates
dates1

pd.date_range(start = '2015-01-01', end = '2016-01-01', freq = 'M') #month end
pd.date_range(start = '2015-01-01', end = '2016-01-01', freq = 'MS') #months start

start, end = '2021-01-01', '2021-02-02'
dates = pd.DataFrame(index = pd.date_range(start, end))   #creating empty dataframe, only the index is defined

df = pd.DataFrame([[10, 20, None, 40],[None, 2, 3, 40]], columns = ['col1', 'col2', 'col3', 'col4'], index = ['row1', 'row2'])
df
df.isnull().any() #if any value in column is True
df.fillna(method = 'ffill')
df.fillna(method = 'bfill')
df.fillna(method = 'ffill', axis = 1)
df.interpolate()
df.replace(np.nan, 'ahoj')

#%% Math Operations
a  = np.random.standard_normal((9, 4))
a = pd.DataFrame(a)
a.round(6)

pd.set_option("precision", 5)
a.round(6)

a // 2 #"floor" divides series

a.rank()
a.rank(ascending = False)

df.describe()

df.transpose()
df.T

df.sum()
df.prod() #product
df.sum(axis = 0) # axis are the same as for numpy 
df.sum(axis = 1)
df['6th'].quantile(.01)
df['6th'].skew()
df['6th'].kurt()
df['6th'].diff() #first difference of a series
df['6th'].clip(lower = 3, upper = 8) #clipped values, changes the min max to the parameters

df.apply(lambda x: x **2)

df ** 2
df + 2
df * 1.0 - 1

numbers.dot(numbers)

df['col2'].mean()
df[['col2', 'numbers']].mean()

np.array(df)   # to generate ndarray from dataframe

a.corr()
a.cov()
a.count()
a.mean()
a.std()
a.cumsum()
a.abs()
a.add(20)
a.nlargest(3, 2) #Return the first `n` rows ordered by `columns` in descending order.
df = np.sqrt(df)
df.sum()

s = pd.Series(np.random.randint(0,10,100))
s

s.head(9)
s+= 100
s    

x = [8.0, 1, 2.5, 4, 28.0]
y = [9.0, 10, 1.5, 40, 28.0]

v = pd.Series(x)
u = pd.Series(y)
v.corr(u)

import math
pd.Series([1.0, math.nan,  4]).mean()  #pandas ignores nan by default X numpy


#%% Plotting

df.cumsum().plot(lw = 2.0)

numbers = pd.Series(data = [1, 22, 44, 65, 89, 75 ,65 , 78] ,name = 'Counts')
numbers2 = pd.Series(data = [1, 23, 44, None, 9, 76 ,65 , 100] ,name = 'Counts2')

import matplotlib.pyplot as plt

fig = plt.figure()
numbers.plot()
numbers2.plot()
plt.legend()
fig.savefig('f:\_Python\General useful\plot.png')

#bar chart
fig = plt.figure()
numbers.plot(kind = 'bar')
numbers2.plot(kind = 'bar', color = 'g', alpha = 0.5)

#histogram
numbers.hist()

numbers.plot(kind = 'kde')
numbers.plot(kind = 'density')

numbers.plot.area() # two ways to specifiy the type of a graph, see below
numbers.plot(kind = 'area') 
numbers.plot.barh()
numbers.plot.box()
numbers.plot(kind = 'pie')


#%% Getting data from yfinance
import yfinance as yf

df_yahoo = yf.download('AAPL', start= '2000-01-01', end = '2010-12-31', actions = 'inline')  #actions: includes stock spkits + dividends

df = df_yahoo.rename(columns = {'Adj Close': 'adj_close'})

df_yahoo.loc['2000-06-21']
df_yahoo.loc[:, :]
df_yahoo.iloc[0:, 0:]
df_yahoo.loc[:, ['Dividends']]
df_yahoo.loc[:, ['Dividends', 'High']]
df_yahoo.loc['2000-06-21', ['Dividends', 'High']]


df['simple_rtn'] = df.adj_close.pct_change()
df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))

index1 = pd.date_range(start = '1999-12-31', end = '2010-12-31')
dates = pd.DataFrame(index = index1)

dates.join(df.adj_close, how = 'left', ).fillna(method = 'ffill')

df = df.asfreq('M')
# alternatively:
df["log_rtn"].resample('M').mean()     # average monthly return
df.asfreq('M') # najde posledni den v mesici a vezme jeho hodnotu, pokud zadna hodnota neni (napr vikend) doplni NaN, proto lepsi pouzit nasledujici:
df = df.resample('M').last() 
    
def realized_vol(x):
    return np.sqrt(np.sum(x**2))

df_rv = df.groupby(pd.Grouper(freq= 'M')).apply(realized_vol)

df_rolling = df['simple_rtn'].rolling(window = 21).agg(['mean', 'std' ])


#%%
csv_path = r'F:\_Python\Tutorial replications\Cookbook\F-F_Research_Data_factors_CSV/F-F_Research_Data_Factors.CSV'

factor_df = pd.read_csv(csv_path, skiprows = 3)
factor_df.columns = ['date', 'mkt', 'smb', 'hml', 'rf']

string = ' Annual Factors: January-December '

indices = factor_df.iloc[:, 0] == string
indices[indices == True]

start_annual = factor_df[indices].index[0]

factor_df = factor_df[factor_df.index < start_annual]
factor_df = factor_df.dropna()

factor_df['date'] = pd.to_datetime(factor_df['date'], format='%Y%m').dt.strftime("%Y-%m")
factor_df = factor_df.set_index('date')

asset = 'FB'
start_date= '2013-12-31'
end_date = '2018-12-31'

factor_df = factor_df.loc[start_date:end_date]
factor_df.info()

# convert to numeric
pd.to_numeric(factor_df.mkt)

factor_df = factor_df.apply(pd.to_numeric)
factor_df.info()
factor_df = factor_df/100

fb_df = yf.download(asset, start = start_date, end = end_date, adjusted = True)
y = fb_df['Adj Close'].resample('M').last().pct_change().dropna()
y.index = y.index.strftime('%Y-%m')
y.name = 'rtn'

ff_data = factor_df.join(y)
ff_data['excess_rtn'] = ff_data['rtn'] - ff_data['rf'] 

y = yf.download('TSLA', start=start_date, end=end_date, adjusted=True, progress=False)
y.index = y.index.strftime('%Y-%m') # excludes day from the index 

y.name = 'return' # rename a series
 
df.dtypes

df.memory_usage(deep=True) / 1024 ** 2 # to be in MB

df.describe().transpose().round(2)
df.describe(include='object').transpose() #describes only categorical
df.describe(include='all')


#%% Dunder https://www.dunderdata.com/blog/selecting-subsets-of-data-in-pandas-part-1

import pandas as pd
import numpy as np

df = pd.DataFrame(data = np.array([['AK', 'blue', 'Apple', 30, 165, 4.6],
                    ['US', 'green', 'Pork', 2, 70, 8.3],
                    ['FL', 'red', 'Mango', 12, 120, 9.0],
                    ['AL', 'white', 'Apple', 4, 80, 3.3],
                    ['AK', 'gray', 'Cheese', 32, 180, 1.8],
                    ['TX', 'black', 'Melon', 33, 172, 9.5],
                    ['TX', 'red', 'Beans', 69, 150, 2.2]], dtype=object) 
             ,columns = pd.Index(['state', 'color', 'food', 'age', 'height', 'score'], dtype=object) # object can be specified with or w/o quotation marks
             ,index = pd.Index(['Jane', 'Niko', 'Aaron', 'Penelope', 'Dean', 'Christina', 'Cornelia'], dtype='object'))

df
df.index
df.columns
df.values #must be declared as "data" when constructing the DF (different from index and columns)

type(df.index)
type(df.columns)
type(df.values)

## SELECTING WITH .LOC or COLUMN LABEL
df['food'] # works only for columns
df.food #not practical....
df[['food', 'color']] # list of columns, not recommended
df[['food', 'food']] # selecting the same column twice
df['Jane': 'Cornelia'] 
df['state': 'food'] # does not work

df[3] #does not work, but a slice would work
df[3:4] # slices work for rows - but are impractical
df['Aaron': 'Christina'] #also works for rows but not recommended for use

type(df['food']) #pandas.core.series.Series
df['food'].index
df['food'].values
df['food'].name # this is the old column name, but series has no columns

# Series has two main components, the index and the data, NO columns in a Series!
df[['color', 'score', 'food']] # list of multiple or just one column
df[['color']] 

# .loc selects by either a columns or row LABEL
df.loc['Niko']
df.loc[['Niko', 'Penelope']]
df.loc['Niko' : 'Penelope']  #includes the last value (as opposed to other python lists etc)
df.loc[:'Penelope']  #other typical slices are possible
df.loc['Penelope':]  
df.loc[:'Penelope':2]

df.loc['food'] #does not work, with .loc, the row parameter is mandatory!
df.loc[:,'food'] 
df.loc['Niko':'Christina':3,'color':'score':2]  #various combinations are possible
df.loc[['Niko','Christina'],'color':'score':2] 

## SELECTING WITH .ILOC (integer location)
# works the same as loc but uses integer location instead of labels
df.iloc[:]
df.iloc[[3, 4]]
df.iloc[1:]
df.iloc[:-1:2]
df.iloc[::2] #all but every second
df.iloc[1:6, [4, 5]] 

#difference between using .iloc and .loc is that iloc excludes the last value, and loc includes it
df.iloc[1:3]
df.loc['Niko' : 'Penelope']

## Series
# does not have columns, so it only selects based on index (row of a dataframe), only one indexer is thus needed
#series has only index and the values

food = df['food']
food.index

food.Jane #selecting by the index - does not work on a DataFrame - there it works only on columns
food['Jane'] #works but better is to use .loc or .iloc

food.loc['Jane']
food.loc['Jane'::2] #various selections are possible
food.loc[['Jane', 'Aaron']]

food.iloc[1] #returns a string
food.iloc[1::2] #returns a series
food.iloc[[1, 5, 2]] 

type(food.iloc[1]) 
type(food.iloc[1::2])

food[[1, 2]] # is possible to select a list of integers as opposed to a python list, see error below
food.tolist()[[1, 2]]
food.tolist()[1:2]
food[1:2] 

# in general its better to use .loc or .iloc
food[2:4] # also not recommended
food['Aaron':'Penelope'] # also not recommended
food[['Aaron','Penelope']] # also not recommended

#new columns
df['new'] = 10

new_series = pd.Series(['Praha', 'Brno', 'Olomouc', 'Dresden', 'Moenchengladbach', 'Berlin', 'Muenster'])
df['Series'] = new_series #Causes NaN!! because the indexes are different! indexes have to match!

df.index == new_series.index

new_series2 = pd.Series(data =['Praha', 'Brno', 'Olomouc', 'Dresden', 
                               'Moenchengladbach', 'Berlin', 'Muenster'],
                        index = ['Jane', 'Niko', 'Aaron', 'Penelope', 'Dean', 'Christina', 'Cornelia']
                        )

df['Series2'] = new_series2 #now it works!

df.score = df.score.astype(int) #changing type of column


#%% save file
import os
directory = 'F:\_Python\General useful'
os.chdir(directory)
file = '\TestFrame.csv'

#1] 
df.to_csv(directory + file, sep = ';')
pd.read_csv(directory + file) # does not work with anything them COMMA separeted files
df.to_csv(directory + file, sep = ',')
pd.read_csv(directory + file) 


#2]
f = open(directory + file, 'w')
f.write(str(df))
f.close()
#os.remove(file)

## Read file
pd.read_csv(directory + file, index_col = 0)
df2 = pd.read_csv(directory + file)
df2.index #if not specified during the import, pandas creates a RangeIndex object

df2.rename(columns= {'Unnamed: 0':'Names'}, inplace = True)

#%% Subsets

directory = 'F:\_Python\General useful'
file = '\QueryResults.csv'

f = pd.read_csv(directory + file)
ten = f.head(10)

ten[[True, False, False,  'a' == 'n' , True, True, False, 1 > -1, 1 == 1 , False]] #list of booleans, points to rows to select

criterion = f['score'] == 5 #assigns booleans series to criterion
f[criterion]
type(f['score'] >= 5) # is pandas series with booleans

#.loc can be used in the exact same way, or column parameters can be added
f.loc[criterion, 'score'::3]

f.shape #number of rows and columns 
f.info()

'''
Only the following operators work with pandas:
& - and (ampersand)
| - or (pipe)
~ - not (tilde)
'''

f[(f.score > 4)  and (f.commentcount > 2)] #does not work (as expected)
f[(f.score > 4) & (f.commentcount > 2)]
f[f.score > 4 & f.commentcount > 2] #does not work

f[(f.score > 4) | (f.commentcount > 2)]
f[(f.score > 4) & ~ (f.commentcount > 2)] #and not
f[~((f.score > 4) & ~ (f.commentcount > 2))] #whole condition wrapped and tilde used

#when there are a lot of OR conditions, two options:
#1]    
criterion = ((f.quest_name == 'Atnhai Atthawit') |
             (f.quest_name ==  'burcak')
             )

f[criterion]

#2]
f[f.quest_name.isin(['Atnhai Atthawit', 'burcak'])] #ISIN

f[f.quest_name.isnull()] #find a column with NaN
f[f.quest_name.isna()] #alias of isnull
f[f.commentcount.between(5, 7)] #between method

#booleans for columns
f[['answercount', 'commentcount']].loc[:,f[['answercount', 'commentcount']].mean() > 1]

a = [1, 2, 3, 4, 5]
a[2:6][0] = 50 # does not work, does not assign anything, python creates temp object
df[df.state=='CA']['color'] = 'red' #SettingWithCopyWarning  thats why I should use .loc
df.state.Jane = 'CA'

df.sort_index()
df.sort_index()['J':'Pe'] #when the index is sorted, I can use indexing by letter...weird?!

# Grouping
df['Quarter'] = ['Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3']
df

groups = df.groupby('Quarter')
groups
groups.mean()
groups.max()
groups.size()

df['Odd_Even'] = ['Odd', 'Even','Odd', 'Even','Odd', 'Even', 'Odd' ]
df
group = df.groupby(['Quarter', 'Odd_Even'])
group.size()
group.mean()

# pivot table
scores = pd.DataFrame({
     'name':['Adam', 'Bob', 'Dave', 'Fred'],
     'age': [15, 16, 16, 15],
    'test1': [95, 81, 89, None],
     'test2': [80, 82, 84, 88],
    'teacher': ['Ashby', 'Ashby', 'Jones', 'Jones']})

scores.pivot_table(index = 'teacher')
scores.pivot_table(index = 'teacher', values = ['test1', 'test2'], aggfunc = 'median')
scores.pivot_table(index = ['teacher', 'age'], values = ['test1', 'test2'], aggfunc = ['median', 'max'])

# melting
pd.melt(scores, id_vars = ['name', 'age'], value_vars = ['test1', 'test2']) 
pd.melt(scores, id_vars = ['name'], value_vars = ['test1', 'test2', 'teacher', 'age'], var_name = 'The Parameter') 

#dummy variable
pd.get_dummies(scores, columns = ['age'])

# joining tables
df1 = pd.DataFrame({'name': ['John', 'George', 'Ringo'], 'color': ['Blue', 'Blue', 'Purple']})
df2 = pd.DataFrame({'name': ['Paul', 'George', 'Ringo'], 'carcolor': ['Red', 'Blue', np.nan]}, index=[3, 1, 2])

pd.concat([df1, df2])
pd.concat([df1, df2], axis = 1)
pd.concat([df1, df2], verify_integrity = True) #throws an error as expected

pd.concat([df1, df2], ignore_index = True) #creates a new index

#join is done joining based on indexes, not columns
df1.set_index('name').join(df2.set_index('name')) #work like left join

#merge method is preferable:
df1.merge(df2) #defualt mode is inner join
df1.merge(df2, how = 'outer') 
df1.merge(df2, how = 'left') 
df1.merge(df2, how = 'right') 


df.describe()
df.describe().to_string(line_width = 200)
print(df.describe().to_string(line_width = 200))




