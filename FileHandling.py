

  
#%% File Handling
"""
"r" - Read - Default value. Opens a file for reading
"a" - Append - Opens a file for appending, creates the file if it does not exist
"w" - Write - Opens a file for writing, creates the file if it does not exist
"x" - Create - Creates the specified file, returns an error if the file exists
"""

f = open("Test.txt", "r")


print(f.read())
l = list(f)
l
 
print(f.read(2))  #number of characters


print(f.readline())

f.close()

print(f.readline())


#write

w = open("Test.txt", 'w')
w.write("ahoj")
w.close()


f1 = open("Test.txt", 'w')

for i in range(10):
    f1.writelines(str(i) + ' ' + str(i ** 2) + '\n')
        
f1.close()    

import os

if os.path.exists("Test.txt"):
    os.remove("Test.txt")


os.rmdir("WorkingFiles")

os.mkdir("WorkingFiles")


























   