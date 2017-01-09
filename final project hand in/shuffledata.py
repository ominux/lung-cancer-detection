import numpy as np
import json
import os 
import pandas as pd



def getPaths():
    posDir="normpos"
    negDir="normneg"
    positive_paths=[os.path.join(posDir,file) for file in os.listdir(posDir)]
    negative_paths=[os.path.join(negDir,file) for file in os.listdir(negDir)]
        
    return positive_paths, negative_paths


           
def saveShuffledSet(outTrain,outValid,negIter):
    
    nb_file=0
    
    pos_paths,neg_paths=getPaths()
    tot_p=len(pos_paths)
    tot_neg=len(neg_paths)
    p=0
    n=0
    negIter
    training_sample=np.array([np.empty([40,40,40])])
    
#    
#    while n<tot_neg:
#          for i in range (n,n+negIter):         
#              with open(neg_paths[i],"r") as infile:
#                  neg_data=np.array(json.load(infile))
#              training_sample=np.append(training_sample,neg_data)
#              
#          with open(neg_paths[i],"r") as infile:
#                  neg_data=np.array(json.load(infile))   
#          training_sample=np.append    
#          n+=negIter
#            
#              
#    nb_neg_examples=neg_data.shape[0]
#    print("neg data shape", neg_data.shape)
#    print("nb of negative examples : ", nb_neg_examples)
#    neg_data_labels=np.zeros((nb_neg_examples),dtype=np.int)
#    
#    
#    neg_sample=np.append(neg_sample,np.array([observations[0,i][2]]),axis=0) 
#    
#    
#    with open(inFilePos,"r") as infile2:
#           pos_data=np.array(json.load(infile2))
#    pos_data=pos_data[:nb_neg_examples,:,:,:]
#
#    print("nb of negative examples : ", pos_data.shape[0])
#    pos_data_labels=np.ones((nb_neg_examples),dtype=np.int)
#    all_labels=np.append(neg_data_labels,pos_data_labels,axis=0)
#    all_data=np.append(neg_data, pos_data,axis=0)
#    print("all labels shape: ",all_labels.shape, "all data shape: ", all_data.shape)      
#    
#    permutation=np.random.permutation(all_labels.shape[0])
#    shuffled_data=all_data[permutation,:,:,:]
#    shuffled_labels=all_labels[permutation] 
#    
#    train_dataset=shuffled_data[:600,:,:,:] 
#    train_labels=shuffled_labels[:600]   
#    
#
#    print("train data set shape",train_dataset.shape)
#    
#    with open(outTrain,"w") as outfile1:
#          saveTrain={
#             "train_dataset":train_dataset.tolist(),
#             "train_labels":train_labels.tolist()
#                     }
#          json.dump(saveTrain,outfile1)
#          
#    print("Training Data Saved")
#    
#    valid_dataset=shuffled_data[600:,:,:,:] 
#    valid_labels=shuffled_labels[600:]   
#    
#    print("validation data set shape",valid_dataset.shape)
#    
#    with open(outValid,"w") as outfile1:
#          saveValid={
#             "valid_dataset":valid_dataset.tolist(),
#             "valid_labels":valid_labels.tolist()
#                     }
#          json.dump(saveValid,outfile1)
#          
#    print("Validation Data Saved")
     

global_count=12600
paths=[]    
 
def generateTrainingSet(inDirectories,outDirectory,label):
         global global_count   
         global paths
         for file in os.listdir(inDirectories):
             print("File being treated: " + file)
             fullpath= os.path.join(inDirectories,file)
             
             with open(fullpath,"r") as infile:
                      data=np.array(json.load(infile))
                                  
             for k in range (data.shape[0]):
                 
                 
                 outName=outDirectory+str(global_count)+".json"
                 
                 with open(outName,"w") as outfile:
                     
                   saveImage={
                     "data":data[k,:,:,:].tolist(),
                     "label":label
                         }
                   json.dump(saveImage,outfile)
                 paths.append(outName)
                 global_count+=1
                 if global_count % 100==0:
                     print(global_count)
                 
         print("Done with this directory, global count is at: ", global_count,  " files")        
             
#generateTrainingSet("normneg","trainingdata/",0)          
generateTrainingSet("normpos","trainingdata/",1)      

print ("total nb of paths: ", len(paths))
paths=pd.DataFrame(paths)    
paths.to_csv("data_for_model_paths.csv",index=False,header=False)        

#saveShuffledSet(outTrain,outValid,3):


