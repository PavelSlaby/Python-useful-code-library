
"""
- Python uses
    - multiparadigm approach -can write code in procedural, object-oriented, functional or imperative manner
    - interpreted
    - Dynamically typed - Types in Python are in general inferred at runtime and not statically declared


- Indentation:
    - The number of spaces is up to you as a programmer, but it has to be at least one. Best practice is to use 4spaces instead of tab.

- PEP 8:
    - a recommendation for a "default" style exists for Python, the PEP 8 (PEP = Python Enchancement Proposal): https://www.python.org/dev/peps/pep-0008/

    Naming Conventions:
        - Modules: all lowercase names
        - Class names: PascalCase
        - Constant variables: SNAKE_CASE
        - function, methods, variables: lowercase snake_case

    - because code is read more often then written, its safer to err on the side of too long variable names

"""

# The Zen of Python
import this

#IPython magics - might not work in all IDEs - ony in IPython
%magic      # all available magic commands, with description
%lsmagic    # short list of the previous
%cd?        # help on a specific magic

# examples of magic commands:
%clear
%history
%timeit  4+4
%time 4 + 4     # Time execution of a Python statement or expression.
%quickref       # Quick reference of all IPython specific syntax and magics.

import keyword
keyword.kwlist  # list of all current reserved keywords in python

?  # -> Introduction and overview of IPython's features.
help()  # Access Python's own help system.
help(str)  # specific help

pip

print(keyword.__doc__) #  Gets docstring from a module/class/function
help(keyword)

print(input("vek:"))




