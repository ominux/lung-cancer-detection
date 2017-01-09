import numpy as np
import tensorflow as tf
import json
import pandas as pd
import time 
from multiprocessing import Pool

#with open('training_set.json', 'r') as t:
#  trFile = json.load(t)
#  train_dataset = np.array(trFile['train_dataset']).astype(np.float32)
#  train_labels = np.array(trFile['train_labels']).astype(np.float32)
#  train_labels=train_labels.reshape(train_labels.shape[0],1)
#  del trFile  #  to help gc free up memory
#  print('Training set', train_dataset.shape, train_labels.shape)
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#with open('validation_set.json', 'r') as v:
#  trFile = json.load(v)
#  valid_dataset = np.array(trFile['valid_dataset']).astype(np.float32)
#  valid_labels = np.array(trFile['valid_labels']).astype(np.float32)
#  valid_labels=valid_labels.reshape(valid_labels.shape[0],1)
#  del trFile  #  to help gc free up memory
#  print('Validation set', valid_dataset.shape, valid_labels.shape)
#
#  


def loadFileintoSets(paths,typeData="Training"):
    
    dataset=np.array([np.empty([40,40,40],dtype=np.float32)])   
    labels= np.array(np.empty([1],dtype=np.float32))  
    
    for path in paths:
#        print(path)
        with open(path, 'r') as v:
          file = json.load(v)
        
        dataset = np.append(dataset, [np.array(file['data']).astype(np.float32)],axis=0)

        labels = np.append(labels, [np.array(file['label']).astype(np.float32)],axis=0)   
        
    dataset=dataset[1:,:,:,:]    
    labels=labels[1:]    
    labels=labels.reshape(labels.shape[0],1)   
    print("total loaded size: " ,labels.shape, " for", typeData)
    
    return dataset, labels




def accuracy(predictions, labels):
 # print("sum of predictions = ", np.sum(predictions))
  #print("sum of labels = ", np.sum(labels))
  return (100.0 * np.sum(predictions == labels)/ predictions.shape[0])


if __name__ == '__main__':
    
    np.random.seed(5)
    """Spliting the data indexes""" 
    df=pd.read_csv("data_for_model_paths.csv",header=None)
    full_data_paths=df.values
    total_nb_data=len(full_data_paths)
    
    image_size = 40
    num_labels = 1
    num_channels = 40 # 3D
      
    batch_size =200
    patch_size = 5
    depth = 80
    num_hidden = 300
        
    nb_valid=100
    nb_training=total_nb_data-nb_valid  
        
    batch_data=np.array([np.empty([40,40,40],dtype=np.float32)]) 
    batch_labels =   labels= np.array(np.empty([1],dtype=np.float32))  
    
    def loadTrainingSets(q, paths):
    
        batch_data=np.array([np.empty([40,40,40],dtype=np.float32)]) 
        batch_labels =  np.array(np.empty([1],dtype=np.float32))
        
        for path in paths:
    #        print(path)
            with open(path, 'r') as v:
              file = json.load(v)
            
            batch_data = np.append(batch_data, [np.array(file['data']).astype(np.float32)],axis=0)
    
            batch_labels = np.append(batch_labels, [np.array(file['label']).astype(np.float32)],axis=0)   
            
        batch_data=batch_data[1:,:,:,:]    
        batch_labels=batch_labels[1:]    
        batch_labels=batch_labels.reshape(batch_labels.shape[0],1)   
        print("total loaded size: " ,batch_labels.shape, " for training batch")
        q.put([batch_data,batch_labels])
    permutation=np.random.permutation(total_nb_data)
    shuffled_data_paths=full_data_paths[permutation]  
      
    
    ##### Size  OF Validation set ! 
    
       
    training_paths=shuffled_data_paths[:nb_training]  
    
    training_paths=training_paths.reshape(training_paths.shape[0],)
    valid_paths=shuffled_data_paths[nb_training:]
    
    valid_paths=valid_paths.reshape(valid_paths.shape[0],)
    
    
    """Importing Validation Dataset , because it doesn't have to be batched everytime 
            and it can be held in memory"""
    
    valid_dataset, valid_labels=loadFileintoSets(valid_paths, " validation set")
    
    print('Validation set', valid_dataset.shape, valid_labels.shape) 
    print("Total Nb positives In Validation: ", valid_labels.sum())
    
    
    ###########################################################
    
    
    graph = tf.Graph()
    
    with graph.as_default():
    
      # Input data.
      tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size,num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      
      # Variables.
      layer1_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, num_channels, depth], stddev=0.1))
      layer1_biases = tf.Variable(tf.zeros([depth]))
      layer2_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth, depth], stddev=0.1))
      layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
      layer3_weights = tf.Variable(tf.truncated_normal(
          [image_size // 8 * image_size // 8 * depth, num_hidden], stddev=0.1))
      layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
      layer4_weights = tf.Variable(tf.truncated_normal(
          [num_hidden, num_labels], stddev=0.1))
      layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
      
      # Model. ; keep_pron is used for dropout, for test it should always be =1
      def model(data,keep_prob):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        pooled=tf.nn.max_pool(conv,ksize=[1,2,2,1],strides=[1, 4, 4, 1],padding='SAME')
        hidden = tf.nn.relu(pooled + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        #adding drop out
        dropped_out=tf.nn.dropout(hidden,keep_prob)
        return tf.matmul(dropped_out, layer4_weights) + layer4_biases
    
      # Training computation.
      logits = model(tf_train_dataset,0.8)
      print("logits shape:", logits.get_shape().as_list())
      print("train_labels shape:", tf_train_labels.get_shape().as_list())
      lossL2 = (tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases)+
                  tf.nn.l2_loss(layer4_weights)+ tf.nn.l2_loss(layer4_biases))*1
      xent=tf.sigmoid(logits)
      loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_train_labels)+lossL2)
        
      # Gradient Coefficient
      global_step = tf.Variable(0,name='global_step',trainable=False)  # count the number of steps taken.
      learning_rate = tf.train.exponential_decay(0.01, global_step, 20 ,0.96)
      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
      #optimizer=tf.train.AdamOptimizer().minimize(loss)
      # Predictions for the training, validation, and test data.
      train_prediction = tf.greater(tf.nn.sigmoid(logits),0.5)
      valid_prediction = tf.greater(tf.nn.sigmoid(model(tf_valid_dataset,1)),0.5)
      
    num_steps = 100
    
    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print('Initialized')
      prev_loss=0
      for step in range(num_steps):
        """ the following offset is used if stochastic grad descent is used, 
        but as I am using whole training data offset is not necessary"""
        
        t1=time.time()
        permut=np.random.permutation(nb_training)
        permut=permut[0:batch_size]
        shuffled_training_paths=training_paths[permut]  
        
        # batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        #batch_labels = train_labels[offset:(offset + batch_size)]
    #    batch_data = train_dataset
    #    batch_labels = train_labels
        
    
        batch_data, batch_labels=loadFileintoSets(shuffled_training_paths, " training set") #The Working one !!!!!
        
        t2=time.time()
        print("Time for Importing: ", batch_size , "training datas:{: .3f} secs".format( t2-t1))
        
        t1=time.time()         
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels:batch_labels}
        _, l, predictions,sigm = session.run(
          [optimizer, loss, train_prediction,xent], feed_dict=feed_dict)
        t2=time.time()
        print("Time for Training : ", batch_size , " datas:{: .3f} secs".format( t2-t1))
        
        if (step %  5==0):
          print('Minibatch loss at step %d: %f' % (step, l))
          #print('Sigmoid function:' ,  np.sum(sigm))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))     
        # Keep track of the training cost
         
          print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction.eval(), valid_labels))  
        prev_loss=loss