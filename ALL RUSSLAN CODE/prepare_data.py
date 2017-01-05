import numpy as np
import os
import scipy.io as sio
import json


#datadir="/Volumes/EXT WD Elements 1 TB/TermProject/termProject/train_data"
#os.listdir(datadir)
   
#with open('dirs.txt', 'w') as outfile:
#    json.dump(arr_dirs, outfile)   



#output: dictionary: key = every path, values = list of all positive ids
def extractSign():
    datadir="/Volumes/EXT WD Elements 1 TB/TermProject/termProject/train_data"
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

#some data analysis

with open("positives.json", "r") as positives_json:
    data = json.load(positives_json)


#rotating images

print data
data['/Volumes/EXT WD Elements 1 TB/TermProject/termProject/train_data/scan_682/candidates.mat']


positive_data = {key:data[key][0] for key in data.keys() if data[key]}
print positive_data


def extrPositvs(posDict,outFile):
    pos_sample=np.array([np.empty([40,40,40])])
    for pth in posDict.keys():
        scanDict=sio.loadmat(pth)
        candids=np.array(scanDict.get('candidates'))
        for cand in candids[0,:]:
            if cand[3]==posDict[pth]:
                pos_sample=np.append(pos_sample,np.array([cand[2]]),axis=0)
                print(pos_sample.shape)
                break #just taking the first observation of a particular scan which is positive
    pos_sample=pos_sample[1:,:,:,:]
    print(pos_sample.shape)
    with open(outFile,"w") as outfile:
        json.dump(pos_sample.tolist(),outfile)




extrPositvs(positive_data, 'outfile.json')


skimage.transform.rotate(image, angle[, ...])



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