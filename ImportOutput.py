
import os
os.getcwd()

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
a.write('just added a line')
a.close()
a = open(file2, 'r')
print(a.read())


a = open(file2, 'w')
a.write('just rewrote the file')

a = open('created_file.txt', 'x')

import os

os.remove('crated_file.txt')

os.path.exists(file)

os.rmdir() # removes a directory
