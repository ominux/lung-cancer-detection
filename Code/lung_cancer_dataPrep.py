
#imports
import scipy.io as sio
import numpy as np
import sys
import os

#loading the data
#information on mat-IO: https://docs.scipy.org/doc/scipy/reference/tutorial/io.html

os.chdir('/Volumes/EXT WD Elements 1 TB/TermProject/termProject') #setting the working directory to the external hard drive

mat_files = sio.loadmat('train_data/scan_3/candidates.mat')

mat_files
mat_files.keys() #keys within the .mat file: ['__version__', 'candidates', '__header__', '__globals__'], only candidates matter!

mat_files_cand = mat_files['candidates']
print mat_files_cand
print mat_files_cand.dtype

mat_files_cand[40]

mat_files_cand[0]




math_files
mat_files[('candidates')] # produces sth interesting






data = np.array(mat_files)
type(data)
print data.shape
print(data[0])



print sio.whosmat('train_data/scan_1/candidates.mat')

os.getcwd() #check the current working directory
os.setcwd

