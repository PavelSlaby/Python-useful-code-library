# %% Math module

import math

math.sqrt(81)
math.pi
math.degrees(math.pi)

math.exp(1)
math.exp(1) == math.e
math.inf  # returns infinity

math.factorial(4)
math.floor(24.2264654)
math.ceil(24.2264654)
math.log(math.exp(1))
math.log(27)

math.log10(100)
math.sin(1)
math.tau

math.fabs(-5.5)

math.nan
math.nan == math.nan

# %% TIME FUNCTIONS
import datetime as dt
import time as tm

tm.time()
tm.time?
tm.time() / (60 * 60 * 24 * 365)  # the year 1970 is the epoch

dtnow = dt.datetime.fromtimestamp(tm.time())
dtnow.month
dtnow.year
dtnow.day
dtnow.strftime('%A')
dtnow.strftime('%B')

a = dt.datetime(2022, 4, 22, 11, 21)
a.day
a.date()
a.time()

a.strftime('%m/%d/%Y %H:%M')
a.strftime('%Y-%m-%d %H:%M')

from datetime import datetime

# Define the past date
past_date = datetime.strptime('2024-07-24', '%Y-%m-%d')

# Get today's date
today = datetime.today()

# Calculate the duration
duration = today - past_date

# Print the duration in days
print("Duration in days:", duration.days)

datetime.strptime??



dt.date.today()
delta = dt.timedelta(days=2)
dt.date.today() - delta

import timeit

print(timeit.timeit('x=(1,2,3,4,5,6,7,8,9,10,11,12)', number=1000000))
print(timeit.timeit('x=[1,2,3,4,5,6,7,8,9,10,11,12]', number=1000000))

%timeit
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

a = dt.datetime.now()
type(a)
print(a)

dt.date(2019, 2, 12)
dir(dt)

dt.date.today().strftime('%y-%m-%d %H:%M')  # two digit year
dt.date.today().strftime('%Y-%m-%d %H:%M')
dt.date.today().strftime('%Y-%m-%d %H:%M %U')  # weeknumber

dt.datetime.now().strftime('%Y-%m-%d %H:%M')

dt.datetime.now().replace(minute=0, second=0).strftime('%Y-%m-%d %H:%M')

tm.perf_counter() - tm.perf_counter()  # how muc time it takes to between the execution of these two functions
tm.sleep(1)  # sleeps

import pandas as pd

pd.date_range(start='2020-01-01', end='2022-01-01', freq='M')
pd.date_range(start='2020-01-01', end='2022-01-01', freq='M').to_pydatetime()

