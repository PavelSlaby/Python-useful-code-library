# idea is to store interesting/new/useful code for future reference
# python:
# - multiparadigm approach -can write code in procedural, object oriented, functional or imperative manner
# - interpreted
# - Dynamically typed - Types in Python are in general inferred at runtime and not statically declared

# %%  Playing with the Code and miscellaneous
# ' tips and tricks: pres Tab after writing a dot after an object to get help
# '''Indentation: The number of spaces is up to you as a programmer, but it has to be at least one. '''

# a recommendation for a "default" style exists for Python, the PEP 8 (PEP = Python Enchancement Proposal)
# https://www.python.org/dev/peps/pep-0008/


import this  # The Zen of Python

%magic  # all available magic commands, with description
%lsmagic  # short list of the previous
# examples of magic commands:
%clear
%history
%timeit
4 + 4
%time
4 + 4  # Time execution of a Python statement or expression.
%%python?
%quickref  # Quick reference of all IPython specific syntax and magics.

import keyword

keyword.kwlist  # list of all curent reserved keywords in python

?  # -> Introduction and overview of IPython's features.
help()  # Access Python's own help system.
help(print)  # specific help

pip
list  # list all the packages installed on your system:

print(input("vek:"))

# In Python, variables are created when you assign a value to it:
x, y, z = "Orange", "Banana", "Cherry"
x = y = z = "Orange"


# using "global" variable - valid outside function
def myfunc():
    global x
    x = "fantastic"
    print(x)


def myfunc2():
    x = "fantastic"
    print(x)


myfunc()
x = 'not-fantastic'
x
myfunc()
x  # x was rewritten by the previous use of the myfunct function
x = 'not-fantastic'
myfunc2()
x  # this time it was not rewritten

try:
    print(10 / 0)
except:  # executed if any error
    print("Yo, error message")

try:
    print(1 / 0)
    print(w)
except ZeroDivisionError:  # executed only if this specific error occurs only
    print("Yo, error: cant divide by zero")

try:
    print(1 / 1)
    print(1 / 0)
except ZeroDivisionError:
    print("error - cant divide by zero, duh ")
else:
    print("its all good")  # printed only if no error occured
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

bool(a)  # but not defined returns error

"""Identity operators are used to compare the objects, 
not if they are equal, but if they are actually the same object, with the same memory location: """

a = 5

a is 5
5 is not "a"
a = 5
b = 5
b is a

''' Data types
Text Type:  	str
Numeric Types: 	int, float, complex
Sequence Types: list, tuple, range
Mapping Type: 	dict
Set Types:  	set, frozenset
Boolean Type: 	bool
Binary Types: 	bytes, bytearray, memoryview
'''

all([1 == 1, 2 == 2, 3 == 4])  # returns true if all is true
any([1 == 1, 2 > 5, 3 == 4])  # returns true if any is true
callable(1)
callable(dir)

~False
~True
bool(~(1 == 2))
bool(~False)

True and False
True and True
True or False
not False
True == True
False == False

x = 5
y = x
id(x)  # identity function, address of the variable in memory, #memory address of the object is same for x and y
id(y)
z = 6
id(z)

a = 990
type(a)
isinstance(a, int)  # checks whether it is the datatype indicated as a second parameter of the function

dir(a)  # all methods and attributes of an object

"""
 List is a collection which is ordered and changeable. Allows duplicate members.
 Tuple is a collection which is ordered and unchangeable. Allows duplicate members.
 Set is a collection which is unordered and unindexed. No duplicate members.
 Dictionary is a collection which is unordered, changeable and indexed. No duplicate members.
"""

# %% Numbers
a = 10
a.bit_length()
b = 1000000000000000000000000000000000000000000000
b.bit_length()
b.denominator

googol_number = 10 ** 100
print(googol_number)

b = 0.35
b + 0.1

c = 0.5
c.as_integer_ratio()
b.as_integer_ratio()  # 15 digit relative accuracy

# to increase accuracy import decimal
import decimal
from decimal import Decimal

decimal.getcontext?
decimal.getcontext()  # default precision is 28, can be changed, see below
# Python itself runs on use the IEEE 754 double-precision standard — i.e., 64 bits   15 digit relative accuracy
decimal.getcontext().prec = 41
Decimal(b) + Decimal(0.1)

1 / 11
Decimal(1) / Decimal(11)

# alternative approach
9 / 3
%precision
10
# floating precision for printing
9 / 3

z = -87.7e100
x = 1j
int(z)
float(10)
int(10.9)
int("1598")
oct(10)
hex(10)
int(True)
int(False)
float(True)
bool(0)

import random

print(random.randrange(1, 3))  # random number between 1 and 2 excluding the last number

15 // 7  # floor division, rounds down automatically

x = 0
x = + 5
x += 5
x -= 4
x *= 10
x /= 10
x **= 3
8 % 7
2 ** 2

3 != 3
3 == 3
3 == 4 or 3 < 5 and not (5 == 7)

min([3, 4])
pow(3, 3)  # power

list(range(1, 10))
abs(-5)

float('nan')

# %% Strings
t = 'this is a string'
t = '''this is 
a string 
over several rows
'''

t.split(" ")
t.split("a")

type(t.split())
t.capitalize()
t.upper()
t.lower()
" Hello, World! ".strip()
t.endswith('g')
t.index('g')

# plenty of more methods...
t.count('s')
t.replace('s', 'cat', 2)

t.find(' ')
t.find('wx')  # -1 there is none

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
print(x + y, end=' XoX ')

"a".replace("a", "b")
"a" in "ahoj"

# There is the old way, via the % character, and the new way, via curly braces ({}) and format().
quantity = 3
itemno = 567
price = 49.95
print("I want to pay %i dollars for %i pieces of item %f." % (quantity, itemno, price))

myorder = "I want to pay {2} dollars for {0} pieces of item {1}."  # newer way then with the % character
print(myorder.format(quantity, itemno, price))

myorder = "I want {:08.2f} pieces of item {} for {} dollars."
print(myorder.format(quantity, itemno, price))

'I have {:08.2f}'.format(2)

C = 10.65484
'this is a variable: %d' % C
print('Value of the European call option: {:.4f}.'.format(C))
'Value of the European call option: {}.'.format(C)
print('Value of the European call option %09.3f.' % C)
'this is a float {:.2f}'.format(15.3456)

"We are the so-called \"Vikings\" from the north."

"center".center(50)
"center".isalpha()
"".isalnum()
"ahoj".find("h")
"center".center(50).strip()
'Aa'.swapcase()

for i in range(10):
    print(i, end='I')

for i in range(10):
    print(i, end='\n')  # default value - line break

# Reverse a string
"Hello World"[::-1]

"Hello World"[1]
"Hello World"[1:2]

"Hello World"[:]

"Hello World"[::2]

"Hello World"[::-2]

# %%
## Tuples
a = [5]
t = (a, 'aba', 12, 1.20)
t = a, 'aba', 12, 1.20  # no need for parentheses
a = (t, 6, 5)
b = (*t, 6, 5)  # astericks "unpacks" the tuple
c = t + (6, 5)  # same effect as unpacking

print(t)
type(t)
t[1]
t[0]
t[0:1][0]
type(t[3])
t.index(12)
t.count(12)

# To create a tuple with only one item, add a comma after the item, otherwise Python will not recognize it as a integer/string.
a = ('a')
a = ('a',)

tupl = (4, 5, 6)
tupl[1] = 6  # expected error

# %% ## Lists
l = [1, 2, 3, 'date']  # 'BRACKET
k = (1, 2, 2, 'date')  # 'PARENTHESES
type(l)
type(k)
k = list(k)
tuple(k)

print(l[2])
l.append([2, 'ahoj'])  # appends by the list
l.append('a')
print(l)
l.extend([2, 'ahoj'])  # extends by the objects in the list

l.insert(1, 'prvni pozice')

l.remove('ahoj')  # removes only the first ocurence
l[2:4]
l[3] = 5

l[1:5:2] = 'gu'  # assigns the sequence on the right to the list on the left, one sign at a time
print(l)

l.count('2')
l.reverse()

list1 = ['a', 'b', 'c']

list1.pop()
list1.pop(0)
list1.pop(2)

list1.index('a')
list1.index('a', 1, 3)

l = [a for a in range(20)]
l[0:20:2] = 'samthingbo'  # replace ever k-th
del l[0:20:2]  # deletes every kth

del list1[1]
del a

list2 = list1.copy()
list3 = list1
list3.clear()

list5 = list2 + list1

list5.sort()
sorted([2, 50, -9, 10])

list5.__len__()

## list comprehensions
m = [i ** 2 for i in range(6)]
m

# remove duplicates from a list
mylist = ["a", "b", "a", "c", "c"]
mylist = list(dict.fromkeys(mylist))

# reverse a list using reversed
for i in reversed([5, 6]):
    print(i)

a = [5, 6]
a.reverse()
a

# %% dicts
d = {
    1: 'A',
    2: 'B',
    3: 'C'
}

c = {
    'country': 'usa',
    'POTUS': 'Donald',
    'status': 'democracy'
}

c['country']
c[['country', 'status']]

list1 = ['country', 'status']
for i in list1:
    print(c[i])

print(d)
type(d)

print(d[1])
print(c['POTUS'])

c.keys()
c.values()
c.items()

c.__getitem__('POTUS')

for key, value in c.items():
    print(key + str(' - ') + value)

c.pop('status')

dicti = dict()
dicti['3'] = '3a'
dicti
dicti.pop('3')
thisdict = dict(brand='Ford', model='F150')
thisdict
dicit2 = dict(dicti)  # how to copy not only dictionaries

dict = ['a', 'b']
enumerate(dict)

for numbers, items in enumerate(dict):
    print(numbers)
    print(items)

c.clear()

# %% Sets
# A set is a collection which is unordered and unindexed

s = set(['u', 'd', 'u'])
print(s)

set1 = {'1', '2', 'a', 'b'}
set1[1]  # error
set1.add('hotel')
set1.update(["orange", "mango", "grapes"])  # to add more than one item
set1.remove('120')  # if the item does not exist, this will raise an error, thats where discard comes in
set1.discard('120')
set1.clear()
set2 = s + set1  # does not work
set2 = s.union(set(['b', 'u']), s)
set2

s.intersection(set2)
set2.difference(s)  # items in set2 but not in set s

u = {'v', 'u'}

u.symmetric_difference(s)  # items in one or the other but not in both

# %%
## Control Structures
a = [1, 2, 3, 4, 5, 6, 7, 8]

# Cycles
for neco in a:
    print(neco)

for i in a[0:4]:
    print(i ** 2)

r = range(0, 8, 1)
type(r)

for i in range(10):
    print(i, end=" ")

for i in range(1, 20):
    if i % 2 == 0:  # % is for modulo
        print("i is even number")
    elif i % 3 == 0:
        print("i is dividable by 3")
    else:
        print("number is odd")

for i in "Hello World":
    print(i)
    if i == "W": break

for i in "Hello World":
    if i == "W": continue  # continues with the next item
    print(i)

for i in range(0, 30, 3):
    print(i)
else:
    print('finito')

if 5 > 4: print(1)

print(True) if 5 > 4 else print(False)
print(True) if 5 > 6 else print(False) if 4 < 2 else print('c')

if 5 > 3: pass  # if statements cannot be empty, put in the pass statement to avoid getting an error.

if 5 > 3:
    print(bool(5 > 3))

# while
total = 0
while total < 100:
    total += 1
    print(total, end=" - ")

i = 0
while i < 6:
    print(i)
    i += 1
    if i == 3: break
    if i == 2:
        continue
    print('2a')
else:
    print('finito')


# %%  Functions

def myfunct(*args):  # if i do not know how many parameters will be passed
    print(args)


# Arbitrary Arguments are often shortened to *args

myfunct(4, 5, a)


# keyword arguments = **kwargs

def my_function(**kid):
    print("His last name is " + kid["lname"])


my_function(fname="Tobias", lname="Refsnes")


def nasob2(x, y, z=None, a=False):
    if z == None:
        return x * y
    elif a:
        return x / y / z
    else:
        return x * y * z


a = nasob2
a(4, 5, 6, 1 == 1)


def power(x):
    return x ** 2


power(2)


def addition(n):
    return n + n


numbers = (1, 2, 3, 4)
k = map(addition, numbers)
print(list(k))


def f():
    return True


if f():
    print("pravda")

# Lambda -  A lambda function can take any number of arguments, but can only have one expression.
x = lambda a: a + 10
x(10)

x = lambda a, c: a ** c if 5 < 3 else c
x(6, 2)


def myfunc(n):
    return lambda a: a * n


mydoubler = myfunc(2)
mytripler = myfunc(3)

print(mydoubler(11))
print(mytripler(11))

# Filter
ages = [5, 12, 17, 18, 24, 32]


def myFunc(x):
    if x < 18:
        return False
    else:
        return True


adults = filter(myFunc, ages)

for x in adults:
    print(x)

adults = filter(myFunc, ages)
list(adults)

# %% Iterators

list1 = [1, 2, 3, 4]
i = iter(list1)
next(i)


# %% Classes, OOP

class FinInst(object):  # simplest class
    pass


FI = FinInst()
type(FI)
FI
FI.price = 100
FI.price  # So-called data attributes — in contrast to regular attributes — can be defined on the fly for every object.


class myClass:
    x = 5


p1 = myClass()
p1.x

"""The self parameter is a reference to the current instance of the class.
It does not have to be named self, but it has to be the first parameter of any function in the class: """


class FI():
    def __init__(self, smbl, prc):
        self.symbol = smbl
        self.price = prc


a = FI('aapl', 600)
a.symbol
a.price
a.prc


class Inst():
    def __init__(self, symbl, prc):
        self.symbol = symbl
        self.__price = prc  # private

    def getprc(self):
        return self.__price


b = Inst('aapl', 10)
b.symbol
b.__price  # private!!
b.getprc()
b._Inst__price  # If the class name is prepended with a single leading underscore, direct access and manipulation are still possible.
b._Inst__price = 500


class FinInstrument(FI):  # inherits FI
    def getprice(self):
        return self.price

    def setprice(self, price):
        self.price = price


inst = FinInstrument('aapl', '800')
inst.symbol
inst.price
inst.getprice()
inst.setprice()
inst.setprice(900)


# example of aggregation
class PortPos():
    def __init__(self, FinInstrument, size):
        self.size = size
        self.position = FinInstrument

    def portval(self):
        return self.size * self.position.getprice()


p = PortPos(inst, 10)
p.size
p.position.getprice()
p.position.symbol
p.portval()


class trida2:
    def __init__(self, prom1, prom2):
        self.v1 = len(prom1)
        self.v2 = prom2

    def myfunct(self):
        print("Hello " + str(self.v1) + " World " + self.v2)


t1 = trida2("bitches", "here i come")

t1.v1  # object properties
t1.v2
t1.myfunct()  # object method

t1.v1 = "Ladies"
t1.v1

trida2.__init__
del t1.v1


# Creating a child class
class t2(trida2):
    pass  # Use the pass keyword when you do not want to add any other properties or methods to the class.


o1 = t2("Show__", "Manship")
o1.v1

# there are more complicated syntax for inheritance.....



