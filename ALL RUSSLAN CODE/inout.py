import numpy as np
import os
import scipy.io as sio
import json
import argparse





def writeOut():
    #datadir="/Volumes/EXT WD Elements 1 TB/TermProject/termProject/train_data"
    datadir = '/Volumes/EXT WD Elements 1 TB/TermProject/termProject/train_data'
    sign={}
    for dir in os.listdir(datadir)[1::]:
        print dir
        """
        fullpath=os.path.join(datadir,dir,"candidates.mat")
        scanDict=sio.loadmat(fullpath)
        observations=np.array(scanDict.get('candidates'))
        cand_ids=np.array([observation[3] for observation in observations[0,:] if observation[0]==1]).tolist()
        sign[fullpath]=cand_ids
        """
    print sign.values()
    return sign

#writeOut()

#how to write to a .mat file
folder_name = 'scan_9'
IDs = np.array([2393857,135346,96325])
predicts = np.array([0.82,0.74,0.56])
labels = np.array([1,1,0])

outPath = '/Users/FabianFalck/ML/lung-cancer-detection/Results/predict_results.mat'
sio.savemat(outPath, {'IDs':IDs, 'predicts':predicts, 'labels': labels})


