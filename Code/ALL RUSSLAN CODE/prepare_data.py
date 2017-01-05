import numpy as np
import os
import scipy.io as sio
import json



   
#with open('dirs.txt', 'w') as outfile:
#    json.dump(arr_dirs, outfile)   

def extractSign():
    datadir="train_data"
    sign={}
    for dir in os.listdir(datadir):
           fullpath=os.path.join(datadir,dir,"candidates.mat")
           scanDict=sio.loadmat(fullpath)
           observations=np.array(scanDict.get('candidates'))
           cand_ids=np.array([observation[3] for observation in observations[0,:] if observation[0]==1]).tolist()
           sign[fullpath]=cand_ids
    return sign 

with open('positives.json', 'w') as outfile:
    json.dump( extractSign(), outfile)   
    
#sign={}
# 
#scanDict=sio.loadmat(arr_dirs[0] )
#candids=np.array(scanDict.get('candidates'))
#for cand in candids[0,:]:
#   #    getting the candidate:
#   if cand[0]==0:
#     sign[str(cand[3])]=dir
#
#with open('positives.json', 'w') as outfile:
#    json.dump(sign, outfile)   
#    
#    
    
    
    
def candD(candidateData):
    return {
       'label':candidateData[0],
        'vol':candidateData[2],
        'cand_id':candidateData[3]
       }



#root=
# data_folders = [
#os.path.join(root, d) for d in sorted(os.listdir(root))
#if os.path.isdir(os.path.join(root, d))]
#  if len(data_folders) != num_classes:
#raise Exception(
#  'Expected %d folders, one per class. Found %d instead.' % (
#    num_classes, len(data_folders)))
#  print(data_folders)
#  return data_folders