import os

os.getcwd()
os.chdir('D:\\_Python')

file1 = 'D:\\_Python\\pokus1.txt'
file2 = r'D:\_Python\pokus2.txt'


folder = 'D:\\_Python'
os.listdir(folder)  # lists all subfolders and files

f = open('D:\_Python\pokus2.txt', "r") #does not work
f = open(r'D:\_Python\pokus2.txt', "r") #works
f = open('D:\\_Python\\pokus2.txt', "r") #works
f = open('D:/_Python/pokus2.txt', "r") #also works

'''
"r" makes it a "raw string literal" -> python will treat \ as an escape character, otherwise it could treat \n as a new line
\\ is also a literla backslash
so python normally treats normal backslash \ as special characters - \n or \t
'''

#or the BEST is to use pathlib - works on windows/mac/linux without changes + easy for joining paths

from pathlib import Path
f = open(Path(r'D:\_Python\pokus2.txt'), "r") # this is one way to do it. Ideally I should close the file, so that other programs can access it.
f.read()
f.close()

# but if something happens in between, the file might still not be accessible to other programs, so a better way is to use with:
with open(r'D:\_Python\pokus2.txt', "r") as f: #it will automatically call the close statement when the execution leaves the block
    print(f.read())

print()


numberOfPets = {'dogs': 2, 'cats': 3}
if 'cats' in numberOfPets:
    print('I have', numberOfPets['cats'], 'cats.', 'eehe')
else:
    print('I have 0 cats.')



'''
 You can pass a string of a folder or filename to Path() to create a Path object of that folder or filename. 
 As long as the leftmost object in an expression is a Path object, you can use the / operator to join together Path objects or strings.
 Compare below
'''

Path('spam') / 'some_folder'
Path('spam') / Path('some_folder')
Path('spam') / Path('some_folder/a')
Path('spam') / Path('some_folder','a')
Path(r'.\spam')

Path.home() # 'home' directory
Path.cwd() #current working directory use oc.chdir() to change it...

folder  = Path(r'D:\_Python')
file = folder / "pokus2.txt"

f = open(file, "r")

print(f.read())
print(f.read(5))  # specify how many characters to return
print(f.readline())  # it iterates

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
a.write('just added a line  ')  # returns the number of signs
a.close()
a = open(file2, 'r')
print(a.read())

a = open(file2, 'w')
a.write('just rewrote the file')
a.close()
a = open(file2, 'r')
print(a.read())


print(a.read())

a = open('created_file.txt', 'x')

a.close()  # file has to be closed before being removed
os.remove('created_file.txt')

os.path.exists(file)

os.rmdir()  # removes a directory

f1 = open("Test.txt", 'w')
for i in range(10):
    f1.writelines(str(i) + ' ' + str(i ** 2) + '\n')

f1 = open("Test.txt", 'r')
f1.read()

f1.close()

if os.path.exists("Test.txt"):
    os.remove("Test.txt")

os.mkdir("WorkingFiles") #creates new directory
os.rmdir("WorkingFiles") #removed directory

# %% Platform
import platform

platform.system()
dir(platform)
platform.uname()
platform.python_version()
platform.processor()
platform.machine()

#pandas
import pandas as pd

sheet1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
sheet2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})


#excel
file_name_xls = 'example.xlsx'
sheet1 .to_excel(file_name_xls, sheet_name='Example1', )

with pd.ExcelWriter(file_name_xls) as writer:
    sheet1.to_excel(writer, sheet_name='Sheet1')
    sheet2.to_excel(writer, sheet_name='Sheet2')


pd.read_excel(file_name_xls, sheet_name='Sheet1', index_col=0, header= None)

#csv
file_name_csv = 'example_csv.csv'
sheet1.to_csv(file_name_csv)
pd.read_csv(file_name_csv)

sheet1.to_csv(file_name_csv, index=False, sep= ',')
pd.read_csv(file_name_csv)

#shutil
import shutil

src = Path(r'D:\_Python\pokus2.txt')
dst = Path(r'D:\_Python\pokus\pokus2.txt')

shutil.move(src, dst)
shutil.move(dst, src) #and lets move it back

shutil.copy(src, dst)

src = Path(r'D:\_Python\pokus2.txt')
src.exists()

