import numpy as np
import json
import scipy.io as sio
from sklearn import preprocessing
import os

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
                print(observ[3], "!!!!!!!!!!!!!!!!!!!! " , pos_sample.shape)
#               print(pos_sample.shape)
                if (nb_positives_so_far%600==0):
                    nb_file+=1
                    with open(outFileName+str(nb_file)+".json","w") as outfile:
                        json.dump(pos_sample[1:,:,:,:].tolist(),outfile) 
                    pos_sample=np.array([np.empty([40,40,40])])
    nb_file+=1                 
    pos_sample=pos_sample[1:,:,:,:]
    print(os.path.join(outFileName,nb_file), " ---- ", pos_sample.shape )  
    with open(outFileName+str(nb_file)+".json","w") as outfile:
                        json.dump(pos_sample[1:,:,:,:].tolist(),outfile)          
                        
 
                                               
                        
                        
def meanNorm(inFile, outFile):
        with open(inFile,"r") as infile:
           data=np.array(json.load(infile))
        print(data.shape)
        
#        min_max_scaler = preprocessing.MinMaxScaler()
#        
#        for n in range(data.shape[0]):
#            for d in range(data.shape[3]):  
#                data[n,:,:,d] =min_max_scaler.fit_transform( data[n,:,:,d]) 
#            
        std=data.std()
        min_nb=data.min()
        mean=data.mean()
        max_nb=data.max()        
        print("min: ", min_nb,"max: ", max_nb," mean: ", mean, "std: ", std)
        print("random example non norm:" ,data[2,2,2,2])
        
        
        meanNormData=np.divide(np.subtract(data,mean),std) 
        std=meanNormData.std()
        min_nb=meanNormData.min()
        mean=meanNormData.mean()
        max_nb=meanNormData.max() 
        print("min: ", min_nb,"max: ", max_nb," mean: ", mean, "std: ", std)
        print("random example  norm:" ,meanNormData[2,2,2,2])
        
    
        with open(outFile,"w") as outfile:
           json.dump(meanNormData.tolist(),outfile) 

           
           
def meanNormPredefined(inFile, outFile,mean_dataset,std_dataset):
        with open(inFile,"r") as infile:
           data=np.array(json.load(infile))
        print(data.shape)

          
        std=data.std()
        min_nb=data.min()
        mean=data.mean()
        max_nb=data.max()        
        print("min: ", min_nb,"max: ", max_nb," mean: ", mean, "std: ", std)
        print("random example non norm:" ,data[2,2,2,2])
        
        
#        meanNormData=np.divide(np.subtract(data,min_nb),(max_nb-min_nb))
        meanNormData=np.divide(np.subtract(data,mean_dataset),std_dataset)  
        std=meanNormData.std()
        min_nb=meanNormData.min()
        mean=meanNormData.mean()
        max_nb=meanNormData.max() 
        
        print("min: ", min_nb,"max: ", max_nb," mean: ", mean, "std: ", std)
        print("random example  norm:" ,meanNormData[2,2,2,2])
   
        with open(outFile,"w") as outfile:
           json.dump(meanNormData.tolist(),outfile)            
           

           
def saveShuffledSet(infileNeg,inFilePos,outTrain,outValid):
    with open(infileNeg,"r") as infile1:
           neg_data=np.array(json.load(infile1))
    nb_neg_examples=neg_data.shape[0]
    print("neg data shape", neg_data.shape)
    print("nb of negative examples : ", nb_neg_examples)
    neg_data_labels=np.zeros((nb_neg_examples),dtype=np.int)
    
    with open(inFilePos,"r") as infile2:
           pos_data=np.array(json.load(infile2))
    pos_data=pos_data[:nb_neg_examples,:,:,:]

    print("nb of negative examples : ", pos_data.shape[0])
    pos_data_labels=np.ones((nb_neg_examples),dtype=np.int)
    all_labels=np.append(neg_data_labels,pos_data_labels,axis=0)
    all_data=np.append(neg_data, pos_data,axis=0)
    print("all labels shape: ",all_labels.shape, "all data shape: ", all_data.shape)      
    
    permutation=np.random.permutation(all_labels.shape[0])
    shuffled_data=all_data[permutation,:,:,:]
    shuffled_labels=all_labels[permutation] 
    
    train_dataset=shuffled_data[:600,:,:,:] 
    train_labels=shuffled_labels[:600]   
    

    print("train data set shape",train_dataset.shape)
    
    with open(outTrain,"w") as outfile1:
          saveTrain={
             "train_dataset":train_dataset.tolist(),
             "train_labels":train_labels.tolist()
                     }
          json.dump(saveTrain,outfile1)
          
    print("Training Data Saved")
    
    valid_dataset=shuffled_data[600:,:,:,:] 
    valid_labels=shuffled_labels[600:]   
    
    print("validation data set shape",valid_dataset.shape)
    
    with open(outValid,"w") as outfile1:
          saveValid={
             "valid_dataset":valid_dataset.tolist(),
             "valid_labels":valid_labels.tolist()
                     }
          json.dump(saveValid,outfile1)
          
    print("Validation Data Saved")
           


    
    

"""Creating negative dataset ; SPECIFY values in the def itself !!!!"""
with open("positives.json","r") as infile:
    paths=json.load(infile)
#neg_paths=[key for key in paths.keys() if not paths[key]]
#extrNegs(neg_paths,"negative_data7.json")

"""Creating Positive dataset ; ceck which value to pick in the def, 0 by default """
pos_example={key:paths[key] for key in paths.keys() if paths[key]}
#extrPositvs(pos_example,"positives.json")
savePositives(pos_example,"posdata/posd")

"""Normalizing"""
#meanNorm('positive_data1.json',"positive_data1_norm_scaled.json" )
#meanNormPredefined('negative_data8.json',"negative_data8_norm_scaled.json",1471.32673,422.8341 )


"""Shuffle and Split Data into Training and VAlidation Sets"""
  
#saveShuffledSet("negative_data8_norm_scaled.json","positive_data1_norm_scaled.json","training_set.json","validation_set.json")



