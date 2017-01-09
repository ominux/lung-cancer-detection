import numpy as np
import json
import os


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

           
           
def meanNormPredefined(inDirectory, outFilePath,label, mean_dataset,std_dataset):
        
      
    nb=0
    for file in os.listdir(inDirectory):    
        
        file=os.path.join(inDirectory,file)
        
        with open(file,"r") as infile:
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
        
        nb+=1
        
        outFileFullName=outFilePath+str(nb)+".json"
        
        # appending a new column indicating the label 
#        if label ==1:
#            label_array=np.ones((meanNormData.shape[0]),dtype=np.float)
#        else:
#            label_array=np.zeros((meanNormData.shape[0]),dtype=np.float)
#
#        meanNormData=np.append(meanNormData,label_array,axis=1)
        
        with open(outFileFullName,"w") as outfile:
           json.dump(meanNormData.tolist(),outfile)            
           
                      
def getParams(inDirectoryPositif, inDirectoryNegatif):
    
    fullData=np.array([np.empty([40,40,40])])
#    i=0
#    imax=3
    for file in os.listdir(inDirectoryNegatif):
#            if i < imax:       
                fullpath=os.path.join(inDirectoryNegatif,file)
                
                with open(fullpath, 'r') as t:
                  dataFile=np.array(json.load(t))
                                 
                print("mean: ", dataFile.mean())  
                print("std: ", dataFile.std())  
                print("min: ", dataFile.min())  
                print("max: ", dataFile.max())  
                
                print("For the whole data !!!!!!!!!!!!:")
                
                fullData=np.append(fullData,dataFile,axis=0)
                print('Full Data size: ', fullData.shape)
                print("mean: ", fullData[1:,:,:,:].mean())  
                print("std: ", fullData[1:,:,:,:].std())  
                print("min: ", fullData[1:,:,:,:].min())  
                print("max: ", fullData[1:,:,:,:].max())  
#            else:
#                 break
#            i+=1    
    for file in os.listdir(inDirectoryPositif):
           
                fullpath=os.path.join(inDirectoryPositif,file)
                
                with open(fullpath, 'r') as t:
                  dataFile=np.array(json.load(t))
                  
                fullData=np.append(fullData,dataFile,axis=0)
                print('Full Data size: ', fullData.shape)
                print("mean: ", fullData[1:,:,:,:].mean())  
                print("std: ", fullData[1:,:,:,:].std())  
                print("min: ", fullData[1:,:,:,:].min())  
                print("max: ", fullData[1:,:,:,:].max()) 
            
             
#getParams("posdata","negdata")

"""Normalizing"""

mean_param=1439
std_param=411

meanNormPredefined("negdata","normneg/norm_neg",0,mean_param,std_param)
meanNormPredefined("posdata","normpos/norm_pos",1,mean_param,std_param)
meanNormPredefined("augmented","normpos/norm_pos_augm",1,mean_param,std_param)