#%% Math module 

import math

math.sqrt(81)
math.pi
math.degrees(math.pi)

math.exp(1)
math.exp(1) == math.e
math.inf #returns infinity

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

#%% TIME FUNCTIONS
import datetime as dt
import time as tm

dtnow = dt.datetime.fromtimestamp(tm.time())
dtnow.month
dtnow.year
dtnow.day
dtnow.strftime('%A')
dtnow.strftime('%B')

dt.date.today()
delta = dt.timedelta(days = 2)
dt.date.today() - delta
   
import timeit 
print(timeit.timeit('x=(1,2,3,4,5,6,7,8,9,10,11,12)', number=1000000))
print(timeit.timeit('x=[1,2,3,4,5,6,7,8,9,10,11,12]', number=1000000))

%timeit x=[1,2,3,4,5,6,7,8,9,10,11,12]

a = dt.datetime.now()
type(a)
print(a)

dt.date(2019, 2, 12)
dir(dt)

dt.date.today().strftime('%Y-%m-%d')

