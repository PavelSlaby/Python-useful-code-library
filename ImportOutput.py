import os
os.getcwd()
os.chdir('f:\\_Python')

file = 'f:\\_Python\pokus.xlsx'
file2 = 'f:\\_Python\pokus.txt'

f = open(file2, "r")

print(f.read())
print(f.read(5)) #specify how many characters to return
print(f.readline()) #it iterates

# to go through the whole file line by line
for i in open(file2, "r"):
    print(i)

f.close()

'''
"r" - Read - Default value. Opens a file for reading, error if the file does not exist
"a" - Append - Opens a file for appending, creates the file if it does not exist
"w" - Write - Opens a file for writing, creates the file if it does not exist
"x" - Create - Creates the specified file, returns an error if the file exists
'''

a = open(file2, 'a')
a.write('just added a line  ') # returns the number of signs
a.close()
a = open(file2, 'r')
print(a.read())

a = open(file2, 'w')
a.write('just rewrote the file')
print(a.read())

a = open('created_file.txt', 'x')

a.close() # file has to be closed before being removed
os.remove('created_file.txt')

os.path.exists(file)

os.rmdir() # removes a directory

f1 = open("Test.txt", 'w')
for i in range(10):
    f1.writelines(str(i) + ' ' + str(i ** 2) + '\n')
     
f1 = open("Test.txt", 'r')     
f1.read()

f1.close()    

if os.path.exists("Test.txt"):
    os.remove("Test.txt")

os.mkdir("WorkingFiles")
os.rmdir("WorkingFiles")


#%% Platform

import platform
platform.system()
dir(platform)
platform.uname()
platform.python_version()
platform.processor()
platform.machine()
