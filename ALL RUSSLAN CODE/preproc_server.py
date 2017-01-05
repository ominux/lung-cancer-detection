import numpy as np
import os
import scipy.io as sio
import json
import argparse

#datadir="/Volumes/EXT WD Elements 1 TB/TermProject/termProject/train_data"
#os.listdir(datadir)

#with open('dirs.txt', 'w') as outfile:
#    json.dump(arr_dirs, outfile)

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-d', '--datadir', type=str)
args = parser.parse_args()
datadir = args.datadir

#output: dictionary: key = every path, values = list of all positive ids
def extractSign():
    #datadir="/Volumes/EXT WD Elements 1 TB/TermProject/termProject/train_data"
    sign={}
    for dir in os.listdir(datadir)[1::]:
            print dir
            fullpath=os.path.join(datadir,dir,"candidates.mat")
            scanDict=sio.loadmat(fullpath)
            observations=np.array(scanDict.get('candidates'))
            cand_ids=np.array([observation[3] for observation in observations[0,:] if observation[0]==1]).tolist()
            sign[fullpath]=cand_ids
    return sign

with open('positives.json', 'w') as outfile:
    json.dump(extractSign(), outfile)