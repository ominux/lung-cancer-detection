import numpy as np
import json
import scipy.io as sio


def savePositives(posDict,outFileName):       
    pos_sample=np.array([np.empty([40,40,40])])
    nb_positives_so_far=0
    nb_file=0
    for pth in posDict.keys():
        scanDict=sio.loadmat(pth)
        observations=np.array(scanDict.get('candidates'))       
        for observ in observations[0,:]:         
            if observ[3] in posDict[pth]:
                nb_positives_so_far+=1
                pos_sample=np.append(pos_sample,np.array([observ[2]]),axis=0) 
#               print(observ[3], "!!!!!!!! " , pos_sample.shape)
#               print(pos_sample.shape)
                if (nb_positives_so_far%600==0):  # dump 600 images into a file and reset the array to empty
                    nb_file+=1
                    print(outFileName+str(nb_file)+".json" , " file is being saved, nb images= ", pos_sample.shape)
                    print("Total number of images saved so far: ", nb_positives_so_far)
                    with open(outFileName+str(nb_file)+".json","w") as outfile:
                        json.dump(pos_sample[1:,:,:,:].tolist(),outfile) 
                    pos_sample=np.array([np.empty([40,40,40])])
   
    print("The last images won't be saved to keep numbers round!!!!!")
#    nb_file+=1                 
#    pos_sample=pos_sample[1:,:,:,:]
#    print(outFileName+str(nb_file)+".json" , " file is being saved, nb images= ", pos_sample.shape) 
#    print("Total number of images saved: ", nb_positives_so_far)
#    with open(outFileName+str(nb_file)+".json","w") as outfile:
#                        json.dump(pos_sample[1:,:,:,:].tolist(),outfile)          
                                                                     
                        
"""Creating negative dataset !!!!"""
with open("positives.json","r") as infile:
    paths=json.load(infile)

"""Creating Positive dataset  """
pos_example={key:paths[key] for key in paths.keys() if paths[key]}


savePositives(pos_example,"posdata/posd")                    