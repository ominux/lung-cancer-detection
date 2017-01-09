import numpy as np
import os
import scipy.io as sio
import pandas as pd
import json

   

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


print('hello')
    
def saveNbOfObservations():  
    datadir="train_data"
    
    paths=[]
    nb_obs=[]
    nb_positives=[]
    for dir in os.listdir(datadir):
        if os.path.isdir(dir):
               fullpath=os.path.join(datadir,dir,"candidates.mat")
               scanDict=sio.loadmat(fullpath)
               observations=np.array(scanDict.get('candidates'))
               cand_shape=observations.shape
               paths.append(dir)
               nb_obs.append(cand_shape[1])
               cand_ids=np.array([observation[3] for observation in observations[0,:] if observation[0]==1]).tolist()
               nb_positives.append(len(cand_ids))
               print(dir)
           
    df=pd.DataFrame({"path":paths, "nb_observs":nb_obs, "nb_positives":nb_positives})

#    print("Min nb of observations:", df["nb_observs"].min())
#    print("Max nb of observations", df["nb_observs"].max())
#    print("Average nb of observations", df["nb_observs"].max())

    return df 
    
    
    
def candD(candidateData):
    return {
       'label':candidateData[0],
        'vol':candidateData[2],
        'cand_id':candidateData[3]
       }


#with open('positives.json', 'w') as outfile:
#    json.dump( extractSign(), outfile)   
#    
#    
#with open('nbObservations.json', 'w') as outfile:
#    json.dump( saveNbOfObservations(), outfile)   
#    
#frame=saveNbOfObservations()    
#frame.to_csv("info_on_data.csv")


#df=pd.read_csv("info_on_data.csv")
#print(df.head())
#print("Total Nuymber of Observations ", df["nb_observs"].sum())
#print("Total Nuymber of Positive Examples ", df["nb_positives"].sum())
#print(df.describe())


sign={}
datadir="train_data"

for dir in os.listdir(datadir):
        pathToScan=os.path.join(datadir,dir)
        if os.path.isdir(pathToScan):   
            fullpath=os.path.join(pathToScan,"candidates.mat")
            scanDict=sio.loadmat(fullpath)
            candids=np.array(scanDict.get('candidates'))
            positives=[]
            for cand in candids[0,:]:
               #    getting the candidate:
               if cand[0]==0:
                   positives.append(cand[3])
                        
            sign[dir]=positives
            print(dir)  
            
with open('positives.json', 'w') as outfile:
    json.dump(sign, outfile)   



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