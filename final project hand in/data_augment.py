import skimage.transform as skif
import json 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os



def normalize(data):
        min_nb=data.min()
        max_nb=data.max()        
        mean=data.mean()        
        std=data.std()        
        print("min: ", min_nb,"max: ", max_nb," mean: ", mean, "std: ", std)
        print("random example non norm:" ,data[2,2,2,2])
        
#        meanNormData=np.divide(np.subtract(data,min_nb),(max_nb-min_nb))
        meanNormData=np.divide(np.subtract(data,min_nb),(max_nb-min_nb))  
        std_after=meanNormData.std()
        min_after=meanNormData.min()
        mean_after=meanNormData.mean()
        max_after=meanNormData.max() 
        print("after normalization :")
        print("min: ", min_after,"max: ", max_after," mean: ", mean_after, "std: ", std_after)
        print("random example  norm:" ,meanNormData[2,2,2,2])
        return meanNormData,max_nb,min_nb


def deNormalize(data,max_nb,min_nb):
        
#        meanNormData=np.divide(np.subtract(data,min_nb),(max_nb-min_nb))
        meanNormData=np.add(np.multiply(data,(max_nb-min_nb)),min_nb) 
        std=meanNormData.std()
        min_nb=meanNormData.min()
        mean=meanNormData.mean()
        max_nb=meanNormData.max() 
        print("after denormalization (should be the same as 1 line before : ")
        print("min: ", min_nb,"max: ", max_nb," mean: ", mean, "std: ", std)
        print("random example  norm:" ,meanNormData[2,2,2,2])
        return meanNormData

       
        
def turn90Deg(inDirectory, outFile):
    
    nb=0
    
    for file in os.listdir(inDirectory):
                
        fullpath=os.path.join(inDirectory,file)
        
        with open(fullpath, 'r') as t:
          data=np.array(json.load(t))
        print('Dataset size', data.shape)
         
        
        normalizedData,max_nb,min_nb=normalize(data)
        
        shape=normalizedData.shape
        
        for n in range (shape[0]):
          for k in range (shape[3]):  
              
            normalizedData[n,:,:,k]=skif.rotate(normalizedData[n,:,:,k],90)
          
        rotatedData=deNormalize(normalizedData,max_nb,min_nb)
        nb+=1
        
        outFileName=outFile+str(nb)+".json"
        print(outFileName, " is being saved !!!!!!!!!!")
        with open(outFileName,"w") as outfile:
                json.dump(rotatedData.tolist(),outfile)
        

def turn180Deg(inDirectory, outFile):
    
    nb=0
    
    for file in os.listdir(inDirectory):
          
        
        fullpath=os.path.join(inDirectory,file)
        
        with open(fullpath, 'r') as t:
          data=np.array(json.load(t))
        print('Dataset size', data.shape)
                 
        normalizedData,max_nb,min_nb=normalize(data)

        shape=normalizedData.shape
        
        for n in range (shape[0]):
          for k in range (shape[3]):  
            normalizedData[n,:,:,k]=skif.rotate(normalizedData[n,:,:,k],180)
          
        rotatedData=deNormalize(normalizedData,max_nb,min_nb)
        nb+=1
        
        
        outFileName=outFile+str(nb)+".json"
        print(outFileName, " is being saved !!!!!!!!!!")
        with open(outFileName,"w") as outfile:
                json.dump(rotatedData.tolist(),outfile)                


def mirror(inDirectory, outFile):
    
    nb=0
    
    for file in os.listdir(inDirectory):
          
        
        fullpath=os.path.join(inDirectory,file)
        
        with open(fullpath, 'r') as t:
          data=np.array(json.load(t))
        print('Dataset size', data.shape)
         
        normalizedData,max_nb,min_nb=normalize(data)
              
        shape=normalizedData.shape
        
        for n in range (shape[0]):
          for k in range (shape[3]):  
#            normalizedData[n,:,:,k]=PIL.ImageOps.mirror(normalizedData[n,:,:,k])
            normalizedData[n,:,:,k]=np.fliplr(normalizedData[n,:,:,k])
          
        rotatedData=deNormalize(normalizedData,max_nb,min_nb)
        nb+=1
        
        outFileName=outFile+str(nb)+".json"
        print(outFileName, " is being saved !!!!!!!!!!")
        with open(outFileName,"w") as outfile:
                json.dump(rotatedData.tolist(),outfile)                  
        
turn90Deg("posdata","augmented/turn90deg")
print(" \n ")
mirror("posdata","augmented/mirror")
print(" \n")
turn180Deg("posdata","augmented/turn180deg")


#fig=plt.figure(figsize=(6,6)) 
#imgplot = plt.imshow(normalizedData[0,:,:,1])
#   
#fig=plt.figure(figsize=(6,6)) 
#imgplot = plt.imshow( skif.rotate(normalizedData[0,:,:,1],90))


