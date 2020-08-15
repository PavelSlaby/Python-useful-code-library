import numpy as np
import pandas as pd

pd.DataFrame([[10, 20, 30, 40],[1, 2, 3, 4]])
df = pd.DataFrame([[10, 20, 30, 40],[1, 2, 3, 4]], columns = ['col1', 'col2', 'col3', 'col4'], index = ['row1', 'row2'])
df

df.index
df.columns

df.sum()
df.sum(axis = 0)
df.sum(axis = 1)


df.apply(lambda x: x **2)

df
df ** 2
df + 2
df * 1.0 - 1

df['5th'] = [3, 2]
df['6th'] = (6, 8)

df['col1']
df

df = df.append({'numbers': 3, '2': 3.1, '3': 5.5, '4': 4.4, '5th': 3.5 ,'6th': 3.4 }, ignore_index = 1)
df1 = df.append({'numbers': 3, '2': 3.1, '3': 5.5, '4': 4.4, '5th': 3.5 ,'6th': 3.4 })


df['col2'].mean()

df[['col2', 'numbers']].mean()

a = np.random.standard_normal((9, 4))
a
a.round(6)

df = pd.DataFrame(a)
df

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
plt.xlabel('date')
plt.ylabel('value')

df

df['Quarter'] = ['Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4', 'Q1']

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
s.head(9)
s+= 100
s    
    



