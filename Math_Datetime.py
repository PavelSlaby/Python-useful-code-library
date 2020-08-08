#%% Math module 

import math

math.sqrt(81)
math.pi
math.degrees(math.pi * 2)

math.exp(1)
math.e
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


#%% TIME FUNCTIONS
import datetime as dt
import time as tm

dtnow = dt.datetime.fromtimestamp(tm.time())
dtnow.month
dt.date.today()
delta = dt.timedelta(days = 2)
today = dt.date.today() - delta
today
    
import timeit 
    print(timeit.timeit('x=(1,2,3,4,5,6,7,8,9,10,11,12)', number=1000000))
    print(timeit.timeit('x=[1,2,3,4,5,6,7,8,9,10,11,12]', number=1000000))

a = dt.datetime.now()
type(a)
print(a)

dt.date(2019, 2, 12)

dir(dt)

a.year
a.day

a.strftime('%A')
a.strftime('%B')





