##%  Playing with the Code and miscellaneous

' tips and tricks: pres Tab after writing a dot after an object to get help

import this

%magic
%lsmagic # alll available magic commands

?
%quickref # Quick reference of all IPython specific syntax and magics.
help() #Access Python's own help system.

pip list #list all the packages installed on your system:

%time 2+5 #    Time execution of a Python statement or expression.

"""
Indentation: The number of spaces is up to you as a programmer, but it has to be at least one. 
"""

aa = input("vek:")

#In Python, variables are created when you assign a value to it: 
x, y, z = "Orange", "Banana", "Cherry"
x = y = z = "Orange" 

def myfunc():
  global x
  x = "fantastic"
  print(x)

myfunc()

o = 5
isinstance(o, int)

try:
    print(w)
    print(10/0)
except:
    print("error vole")
    
    
try:
    print(1/0)
   # print(w)
except ZeroDivisionError:
    print("error: cant divide by zero")
      
try:
    print(1/1)
    print(1/0)
except ZeroDivisionError:
    print("error - cant divide by zero, duh ")
else:
    print("its all good")
finally:
    print("does not matter if it was error or not")    
    
    
if 1 == 1:
    raise Exception("equality")  
       
if 2 == 2:
    raise TypeError("Type error equality")  
    
"""
Any string is True, except empty strings.
Any number is True, except 0.
Any list, tuple, set, and dictionary are True, except empty ones.
"""

bool(a)


"""Identity operators are used to compare the objects, 
not if they are equal, but if they are actually the same object, with the same memory location: """

4 is 5
5 is not "a"
a = 5
b = a
b is a

"""
 List is a collection which is ordered and changeable. Allows duplicate members.
 Tuple is a collection which is ordered and unchangeable. Allows duplicate members.
 Set is a collection which is unordered and unindexed. No duplicate members.
 Dictionary is a collection which is unordered, changeable and indexed. No duplicate members.
"""

# a recommendation for a "default" style exists for Python, the PEP 8 (PEP = Python Enchancement Proposal)

a = 990
type(a)
a.bit_length()

dir(a) #all methods and attributes of an object

''' Data types
Text Type:  	str
Numeric Types: 	int, float, complex
Sequence Types: list, tuple, range
Mapping Type: 	dict
Set Types:  	set, frozenset
Boolean Type: 	bool
Binary Types: 	bytes, bytearray, memoryview
'''

#%% Numbers 
googol_number = 10 ** 100
print(googol_number)

b = 0.35
b + 0.1

c = 0.5
c.as_integer_ratio()
b.as_integer_ratio()   # 15 digit relative accuracy

#to increase accuracy import decimal
import decimal
from decimal import Decimal

import matplotlib
matplotlib?

decimal?  #is apparently stored in a different folder than matplotlib
decimal.getcontext?
decimal.getcontext()

1/11
Decimal(1) / Decimal(11)

decimal.getcontext().prec = 21
Decimal(1) / Decimal(11)

# alternative approach
9 /3    
%precision 10
# floating precision for printing
9 /3

z = -87.7e100
x = 1j
int(z)
float(10)
int(10.9)
int("1598")

import random
print(random.randrange(1, 10))   # random number between 1 and 9

15 // 7  #floor division, rounds down automatically 

x = 0
x =+ 5
x += 5
x -= 4
x *= 10
x /= 10
x **= 3
8 % 7
2 **2

3 != 3
3 == 3

3 == 4 or 3 <5 and not(5==7)

min([3,4])
pow(3, 3) #power

list(range(1, 10)) 

#%% Strings
t = 'this is a string'
t.split(" ") 

type(t.split())
t.capitalize()
t.upper()
t.lower()
" Hello, World! ".strip()
t.endswith('g')
t.index('g')

# plenty of more methods

t.count('s')
t.replace('string', 'cat')

t.find(' ')

'http://www.python.org/'.strip('http:/')

MyTuple = ('-a', 'b', 'c-')
hyphen = '-'
c = hyphen.join(MyTuple)
print(c)
c.strip('-')

a = """ 
Lorem 
ipsum 
dolor
"""

a[1:10]
print(a[1:10])

x = "Python is " 
y = "awesome"
print(x + y)

"a".replace("a", "b")
"a" in "ahoj"

quantity = 3
itemno = 567
price = 49.95
myorder = "I want to pay {2} dollars for {0} pieces of item {1}."
print(myorder.format(quantity, itemno, price))

quantity = 3
itemno = 567
price = 49.95
myorder = "I want {} pieces of item {} for {} dollars."
print(myorder.format(quantity, itemno, price))

"We are the so-called \"Vikings\" from the north."

"center".center(50)
"center".isalpha()
"".isalnum()
"ahoj".find("h")
"center".center(50).strip()


#%%
## Tuples
a = [5]
t = (a, 'aba', 12, 1.20)
t = a, 'aba', 12, 1.20
print(t)
type(t)
t[1]
t[0]
t[0:1]
type(t[3])
t.index(12)
t.count(12)

#To create a tuple with only one item, you have to add a comma after the item, otherwise Python will not recognize it as a integer.
a = ('a')
a = ('a',)

tupl = (4, 5, 6)
tupl[1] = 6

#%% ## Lists
l = [1, 2, 3, 'date']  # 'BRACKET
k = (1, 2, 3, 'date')  # 'PARENTHESES
type(l)
type(k)

k = list(k)

print(l[2])
l.append([2, 'ahoj'])
print(l)
l.extend([2, 'ahoj'])

l.insert(1, 'prvni pozice')

l.remove('ahoj') #removes only the first ocurence
l[2:4]
l[3] = 5

l[1:5:2] = 'gu' #assigns the sequence on the right to the list on the left, one sign at a time
print(l)

l.count('2')
l.reverse()

list1 = ['a', 'b', 'c']

list1.pop()
list1.pop(1)

del list1[1]
del a

list2 = list1.copy()
list3 = list1
list3.clear()

list4 = list(list1)

list5 = list2 + list1
list((1,2,5))
list5.reverse()
list5.sort()

list5.__len__()

list1.clear()

## list comprehensions
m = [i ** 2 for i in range(6)]   
m
    
#%% dicts
d = {
     1 : 'A',
     2 : 'B',
     3 : 'C'
     }

c = {
     'country' : 'usa',
     'POTUS' : 'Donald',
     'status' : 'hovado'
     }

print(d)
type(d)

print(d[2])
print(c['POTUS'])

c.keys()
c.values()
c.items()

c.__getitem__('POTUS')

dict = {'a': 'aloha', 'b': 'baloha', 'c': 'caloha'}
    dict.values()
    dict.items()
    dict.keys()    
  
for key, item in dict.items():
    print(key + str(' - ' ) + item)
    
dict.pop('c') 
    
dicti = dict()
dicti['3'] = '3a'
dicti
dicti.pop('3')
thisdict = dict(brand = 'Ford', model = 'F150')
thisdict
dicit2 = dict(dicti) #how to copy not only dictionaries


dict = ['a', 'b']
enumerate(dict)
    
for numbers,items in enumerate(dict):
    print(items)
    print(numbers)

#%% Sets
#A set is a collection which is unordered and unindexed

s = set(['u', 'd', 'u'])
print(s)

set1 = {'1' , '2', 'a', 'b'}
set1[1]  # error
set1.add('hotel')
set1.update(["orange", "mango", "grapes"]) # to add more than one item
set1.remove('120') # if the item does not exist, this will raise an error, thats where discard comes in
set1.discard('120')
set1.clear()
set2 = s + set1 # does not work
set2 = s.union(set1, set2)
set2

s.intersection(set2)


#%%
## Control Structures

a = [1, 2, 3, 4, 5, 6, 7, 8]

# Cycles
for neco in a:
    print(neco)    

for i in a[0:4]:
    print(i**2)

r = range(0, 8, 1)
type(r)

for i in range(2,6):
    print(i)
    
for i in range(10): 
    print(i, end = " ") 

for i in range(1,20):
    if i % 2 == 0: # % is for modulo
        print("i is even number")
    elif i % 3 == 0:
        print("i is dividable by 3")
    else: print("number is odd")

for i in "Hello World":
    print(i)
    if i == "W": break
    
    
for i in "Hello World":
    if i == "W": continue  #continues with the next item
    print(i)  
    
for i in range(0, 30, 3): print(i) 
else: print('finito') 

if 5 > 4: print(1)

print(True) if 5 > 4 else print(False)
print(True) if 5 > 6 else print(False) if 4 < 2 else print('c')

if 5 > 3: pass #if statements cannot be empty, put in the pass statement to avoid getting an error.

# while

total = 0
while total < 100:
    total += 1
    print(total, end = " - ")        


i = 0
while i < 6:
    print(i)
    i +=1
    if i == 3: break
    if i == 2: 
        continue 
    print('2a')
else:   
    print('finito')
    
#%%  Functions

def myfunct(*args):  #if i do not know how many parameters will be passed
    print(args)

#Arbitrary Arguments are often shortened to *args

myfunct(4,5, a)

#keyword arguments = kwargs

def my_function(**kid):
  print("His last name is " + kid["lname"])

my_function(fname = "Tobias", lname = "Refsnes") 


def nasob2 (x, y, z=None, a = False):
    if z == None:
        return x * y
    elif a:
        return x / y / z
    else:
        return x * y * z
    
a = nasob2 
a(4,5,6, 1==1)

def power(x):
    return x ** 2

power(2)

def addition(n): 
    return n + n 

numbers = (1, 2, 3, 4) 
k = map(addition, numbers) 
print(list(k))

import math

def odm(x):
    return math.sqrt(x) == 3

odm(10)

def f():
    return True

if f():
  print("pravda")
  
# Lambda -  A lambda function can take any number of arguments, but can only have one expression.
x = lambda a : a + 10
x(10)

x = lambda a,c: a** c
x(6, 2)

def myfunc(n):
  return lambda a : a * n

mydoubler = myfunc(2)
mytripler = myfunc(3)

print(mydoubler(11))
print(mytripler(11))

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

#%% Iterators

list1 = [1, 2, 3, 4 ]
i = iter(list1)
next(i)  

#%% Modules

import os
os.getcwd()
os.chdir("f:\_Python")

import MyModule as mm

mm.greet("Pavlos")
dir(mm)


#%% Classes

class myClass:
    x = 5
    
p1 = myClass()
p1.x

class trida:
    def __init__(self, name, age):
      self.name1 = name
      self.age1 = age
        
p1 = trida("Ja", 27)

p1.name1
p1.age1 


"""The self parameter is a reference to the current instance of the class.
It does not have to be named self, but it has to be the first parameter of any function in the class: """

class trida2:
    def __init__(self, prom1, prom2):
        self.v1 = len(prom1)  
        self.v2 = prom2
        
    def myfunct(self):
        print("Hello " + str(self.v1) + " World " + self.v2)

t1 = trida2("bitches", "here i come")

t1.v1 # object properties
t1.v2
t1.myfunct() # object method


t1.v1 = "Ladies"
t1.v1

trida2.__init__
del t1.v1

#Creating a child class

class t2(trida2):
    pass  #Use the pass keyword when you do not want to add any other properties or methods to the class.

o1 = t2("Show__", "Manship")

o1.v1

    
class Student(trida2):
    def __init__(self, fname, lname):
        self.firstname = fname
        self.lastname = lname
        
a = Student("Pavlos", "Hejhoo")        
a.firstname        
a.lastname        
        
#there are more complicated syntax for inheritance.....
    