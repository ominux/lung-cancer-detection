
#imports
import scipy.io as sio
import sys
import os

#loading the data
#information on mat-IO: https://docs.scipy.org/doc/scipy/reference/tutorial/io.html

sys.path.insert(0, '/Volumes/EXT WD Elements 1 TB/TermProject/termProject')

mat_files = sio.loadmat('train_data/scan_1/candidates.mat')

mat_files = sio.loadmat('candidates.mat')




os.getcwd()
os.setcwd

os.chdir('/Volumes/EXT WD Elements 1 TB/TermProject/termProject') #setting the working directory to the external hard drive