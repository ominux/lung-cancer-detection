import numpy as np
import json
import scipy.io as sio



def saveNegatives(paths,outFileName):

    neg_sample=np.array([np.empty([40,40,40])])
    nb_negatives_so_far=0
    nb_file=0
    pos_dict={key:paths[key] for key in paths.keys() if paths[key]}
    for pth in paths.keys():
        scanDict=sio.loadmat(pth)
        observations=np.array(scanDict.get('candidates'))  
        totalImagesInScan=len(observations[0,:])
        print(totalImagesInScan)
        if pth not in pos_dict.keys(): #if the whole scan is negative: 
            print("Current Scan has only negatives: ", pth)
            i=0
            while i<totalImagesInScan:
               neg_sample=np.append(neg_sample,np.array([observations[0,i][2]]),axis=0) 
#               print(observations[0,i][3], "!!!!!!!!! " , neg_sample.shape)
               nb_negatives_so_far+=1
               if (nb_negatives_so_far%600==0):
                        nb_file+=1
                        print(outFileName+str(nb_file)+".json" , " file is being saved, nb images= ", neg_sample.shape)
                        print("Total number of images saved so far: ", nb_negatives_so_far)
                        with open(outFileName+str(nb_file)+".json","w") as outfile:
                            json.dump(neg_sample[1:,:,:,:].tolist(),outfile) 
                        neg_sample=np.array([np.empty([40,40,40])])
               i+=4                   
        else:   
            print("Current scan contains positives: ", pth)
            i=0
            while i<totalImagesInScan:
                if observations[0,i][3] not in pos_dict[pth]:
                    nb_negatives_so_far+=1
                    neg_sample=np.append(neg_sample,np.array([observations[0,i][2]]),axis=0) 
#                    print(observations[0,i][3], "!!!!!!!!! " , neg_sample.shape)
    #               print(neg_sample.shape)
                    if (nb_negatives_so_far%600==0):
                        nb_file+=1
                        print(outFileName+str(nb_file)+".json" , " file is being saved, nb images= ", neg_sample.shape)
                        print("Total number of images saved so far: ", nb_negatives_so_far)
                        with open(outFileName+str(nb_file)+".json","w") as outfile:
                            json.dump(neg_sample[1:,:,:,:].tolist(),outfile) 
                        neg_sample=np.array([np.empty([40,40,40])])
                i+=6
    
    print("The last images won't be saved to keep numbers round !")
    
#    nb_file+=1  
#    neg_sample=neg_sample[1:,:,:,:]
#    print(outFileName+str(nb_file)+".json" , " file is being saved, nb images= ", neg_sample.shape)   
#    print("Total number of images saved: ", nb_negatives_so_far)
#    with open(outFileName+str(nb_file)+".json","w") as outfile:
#                        json.dump(neg_sample[1:,:,:,:].tolist(),outfile)     
                        
                        
"""Creating negative dataset ; SPECIFY values in the def itself !!!!"""
with open("positives.json","r") as infile:
    paths=json.load(infile)

"""Generating negative dataset  """
#pos_example={key:paths[key] for key in paths.keys() if paths[key]}
saveNegatives(paths,"/negdata/negd")
    
#saveNegatives(pos_example,"negdata/negd")                                        