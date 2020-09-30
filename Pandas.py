import numpy as np
import pandas as pd

pd.DataFrame([[10, 20, 30, 40],[1, 2, 3, 4]])
df = pd.DataFrame([[10, 20, 30, 40],[1, 2, 3, 4]], columns = ['col1', 'col2', 'col3', 'col4'], index = ['row1', 'row2'])
df

df.index
df.columns
df.values

df.sum()
df.sum(axis = 0) # axis are the same as for numpy 
df.sum(axis = 1)

df.apply(lambda x: x **2)
np.ones(10).apply(lambda x: x **2) # does not work on numpy
list(map(lambda x: x + 89, np.ones(10))) # map works on numpy

df
df ** 2
df + 2
df * 1.0 - 1

df['5th'] = [3, 2]
df['6th'] = (6, 8)

df['col1']
df

df = df.append({'numbers': 3, '2': 3.1, '3': 5.5, '4': 4.4, '5th': 3.5 ,'6th': 3.4 }, ignore_index = 1) #replaces the index!!! and creates a new row if it does not know to which of existing rows to assging the new data
df1 = df.append({'numbers': 3, '2': 3.1, '3': 5.5, '4': 4.4, '5th': 3.5 ,'6th': 3.4 }) #does not know to which row to assign
df1 = df.append(pd.DataFrame({'col1': 3, 'col2': 3.1, 'col3': 5.5, 'col4': 4.4,}, index = ['row3'])) #remedies the problem above
df2 = df.append(pd.DataFrame({'col1': 3, 'col2': 3.1}, index = ['row3'])) #incomplete row results in NaN
df2.values #np array object 

df['col2'].mean()

df[['col2', 'numbers']].mean()

a = np.random.standard_normal((9, 4))
a
a.round(6)

df = pd.DataFrame(a)
df

type(df[0])
df[0].name
df[0].dtype
df[0].index

df2 = pd.DataFrame(a, index = range(11, 20), columns = ('a', 'b', 'c', 'd'))
df2

df.columns = ['1st' ,'2nd', '3rd', '4th']
df['2nd'][2]

dates = pd.date_range('2015-01-01,', periods = 9, freq = 'M')
dates1 = pd.date_range(start = '2015-01-01', end = '2016-01-01', freq = 'M')
dates
dates1

df.index = dates
df

pd.date_range(start = '2015-01-01', end = '2016-01-01', freq = 'M') #month end
pd.date_range(start = '2015-01-01', end = '2016-01-01', freq = 'MS') #months start

df
np.array(df)   # to generate ndarray from dataframe

df.sum()
df.mean()
df.std()
df.cumsum()
df.describe()
df.abs()
df.add(20)
df.nlargest(3, '1st')
df = np.sqrt(df)
df.sum()

df.cumsum().plot(lw = 2.0)

type(df['1st'])

import matplotlib.pyplot as plt

df['1st'].cumsum().plot(style = 'r', lw = 2.0)

df['Quarter'] = ['Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4', 'Q1']
df

groups = df.groupby('Quarter')
groups
groups.mean()
groups.max()
groups.size()

df['Odd_Even'] = ['Odd', 'Even','Odd', 'Even','Odd', 'Even', 'Odd', 'Even', 'Odd', ]
df
group = df.groupby(['Quarter', 'Odd_Even'])
group.size()
group.mean()

sports = {'NBA':'Basketball', 'NFL':'Football', 'MLB':'Baseball', 'NHL':'Hockey'}
pd.Series(sports)

s = pd.Series(np.random.randint(0,10,100))
s
s.iloc[0]
s.loc[0]

s.head(9)
s+= 100
s     



