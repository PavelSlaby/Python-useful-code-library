# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 22:09:51 2022

@author: pavel slaby

overview of decorators

"decorator is a function that takes another function and extends the behavior 
of the latter function without explicitly modifying it."

"""


def say_hello(name):
    return f"Hi {name}"

def greet_smb(greeting_function):
    return greeting_function("Pavel")


say_hello('Petr')
greet_smb(say_hello) 
'''
passing a function to another function, when naming a function without 
parentheses only reference to the function is passed. 
If there are parentheses the function is called
'''

# inner functions
def parent_f(child):
    print("calling the parent() function")

    def first_child_f():
        print("calling the first_child() function")

    def second_child_f():
        print("calling the second_child() function")

    if child == 2:
        return second_child_f() #the function will be called
    else:   
        return first_child_f #only reference to the functio will be returned

parent_f(1)
first = parent_f(1)
first()

parent_f(2)
first = parent_f(2)
first() #print does a side effect and returns a NoneType and NoneType is not callable
second_child_f  # throws not defined error - can not be called from outside

# decorators

def decorator1(func):
    def wrapper():
        print("print 1")
        func()
        print("print 2")
    return wrapper #does not call anything, just returns the reference!!

def say_hi():
    print("hi!")


decorator1(say_hi) 
decorator1(say_hi)() #is called

say_hi = my_decorator(say_hi) #using the decorator
say_hi
say_hi()

# the pie syntax with @:   
def decorator1(func):
    def wrapper():
        print("ex2 print 1")
        func()
        print("ex2 print 2")
    return wrapper #does not call anything, just returns the reference!!

@decorator1
def say_hi():
    print("hi!")

say_hi()


def twice_decorator(func):
    def wrapper(*args): #use *args and **kwargs if there are arguments to be passed
        func(*args)
        func(*args)
    return wrapper

@twice_decorator
def say_hi(word):
    print(word)

say_hi('say hey')


# measures time
def timing_func(func):
    import time
    def wrapper(*args):
        start = time.perf_counter()
        
        func(*args)
        
        end = time.perf_counter()
        
        time_dif = end - start
        return time_dif
    
    return wrapper

@timing_func
def time_sleep(t):
        time.sleep(t)


time_sleep(3)



# decorators using classes

class decorator_class():
   def __init__(self, function):
       self.function = function

   def __call__(self, *args):
       product = args[0] * args[1]
       print("Product of {} and {} is {} ".format(args[0], args[1], product))
       return self.function(args[0], args[1])


def add(num1, num2):
   value = num1 + num2
   print("The sum of {} and {} is {}.".format(num1, num2, value))

decorator_object = decorator_class(add)
decorator_object(10, 20)






