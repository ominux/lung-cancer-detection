import numpy as np
import json
import scipy.io as sio

def extrNegs(negPath,outfile):
    neg_sample=np.array([np.empty([40,40,40])])
    for pth in negPath:
        scanDict=sio.loadmat(pth)
        candids=np.array(scanDict.get('candidates'))
        slice1=np.array([candids[0,7][2]])
        slice2=np.array([candids[0,14][2]])
        slice3=np.array([candids[0,21][2]])
        slice4=np.array([candids[0,28][2]])
        neg_sample=np.append(neg_sample,slice1,axis=0)
        neg_sample=np.append(neg_sample,slice2,axis=0)
        neg_sample=np.append(neg_sample,slice3,axis=0)
        neg_sample=np.append(neg_sample,slice4,axis=0)
    neg_sample=neg_sample[1:,:,:,:]
    print(neg_sample.shape)
    with open(outfile,"w") as outfile:
        json.dump(neg_sample.tolist(),outfile)
        

def extrPositvs(posDict,outFile):
    pos_sample=np.array([np.empty([40,40,40])])
    for pth in posDict.keys():
        scanDict=sio.loadmat(pth)
        candids=np.array(scanDict.get('candidates'))
        for cand in candids[0,:]:
          if cand[3]==posDict[pth]:
            pos_sample=np.append(pos_sample,np.array([cand[2]]),axis=0)
            print(pos_sample.shape)
            break
    pos_sample=pos_sample[1:,:,:,:]
    print(pos_sample.shape)
    with open(outFile,"w") as outfile:
        json.dump(pos_sample.tolist(),outfile)        
        
                                          
    
  

"""Creating negative dataset ; SPECIFY values in the def itself !!!!"""
#with open("positives.json","r") as infile:
#    paths=json.load(infile)
#neg_paths=[key for key in paths.keys() if not paths[key]]
#extrNegs(neg_paths,"negative_data7.json")


"""Normalizing"""
#meanNorm('positive_data1.json',"positive_data1_norm_scaled.json" )
#meanNormPredefined('negative_data8.json',"negative_data8_norm_scaled.json",1471.32673,422.8341 )


"""Shuffle and Split Data into Training and VAlidation Sets"""
  
#saveShuffledSet("negative_data8_norm_scaled.json","positive_data1_norm_scaled.json","training_set.json","validation_set.json")



