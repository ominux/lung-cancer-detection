import numpy as np
import tensorflow as tf
import json
import pandas as pd
import time 






""" IMPORTANT !!!!!!! """
""" NOrmalize the Data before running it """

""" mean =1439 ; std = 411 ;  meanNormData=np.divide(np.subtract(data,mean),std)   """ 



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


np.random.seed(5)
"""Spliting the data indexes""" 
df=pd.read_csv("data_for_model_paths.csv",header=None)
full_data_paths=df.values
total_nb_data=len(full_data_paths)

image_size = 40
num_labels = 1
num_channels = 40 # 3D
  
batch_size =600
patch_size = 5
depth =120
num_hidden = 300
num_hidden2=160    
nb_valid=800
nb_training=total_nb_data-nb_valid  
    
batch_data=np.array([np.empty([40,40,40],dtype=np.float32)]) 
batch_labels =   labels= np.array(np.empty([1],dtype=np.float32))  

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
      [num_hidden, num_hidden2], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))
  
  layer5_weights = tf.Variable(tf.truncated_normal(
      [num_hidden2, num_labels], stddev=0.1))
  layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
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
    hidden = tf.nn.relu(tf.matmul(hidden, layer4_weights) + layer4_biases)        
    dropped_out=tf.nn.dropout(hidden,keep_prob)
    return tf.matmul(dropped_out, layer5_weights) + layer5_biases
   
  # Training computation.
  logits = model(tf_train_dataset,0.9)
  print("logits shape:", logits.get_shape().as_list())
  print("train_labels shape:", tf_train_labels.get_shape().as_list())
  lossL2 = (tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases)+
              tf.nn.l2_loss(layer4_weights)+ tf.nn.l2_loss(layer4_biases))*0.2
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
  
  saver=tf.train.Saver()
 
  
 
num_steps = 200

#path='model/'
#ckpt_name = 'model.ckpt'
#fname = 'model.tfmodel'
#dst_nodes = ['Y']

with tf.Session(graph=graph) as session:
  
#  saver = tf.train.Saver() 
  tf.initialize_all_variables().run()
  print('Initialized')
  prev_loss=0
  
#      if os.path.exists("models/model.ckpt"):
#        saver.restore(session, "model.ckpt")
#        print("Model restored.")
    
 
  t1=time.time()
  permut=np.random.permutation(nb_training)
  shuffled_training_paths=training_paths[permut[0:batch_size]]  
    
    # batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    #batch_labels = train_labels[offset:(offset + batch_size)]
    #    batch_data = train_dataset
    #    batch_labels = train_labels
    
    
  batch_data, batch_labels=loadFileintoSets(shuffled_training_paths, " training set") #The Working one !!!!!

  print("Time for Importing: ", batch_size , "training datas:{: .3f} secs".format( time.time()-t1))  

  for step in range(num_steps):
    """ the following offset is used if stochastic grad descent is used, 
    but as I am using whole training data offset is not necessary"""
    
        
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels:batch_labels}
    _, l, predictions,sigm = session.run(
      [optimizer, loss, train_prediction,xent], feed_dict=feed_dict)
    
    
    if (step %  8==0):     
      permut=np.random.permutation(nb_training)
      shuffled_training_paths=training_paths[permut[0:batch_size]]  
      batch_data, batch_labels=loadFileintoSets(shuffled_training_paths, " training set") #The Working one !!!!!  
      print('Minibatch loss at step %d: %f' % (step, l))
      #print('Sigmoid function:' ,  np.sum(sigm))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))     
    # Keep track of the training cost
     
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))  
      saver.save(session, 'my-model', global_step=step)
    prev_loss=loss
    
    
#        saver.restore(session, ckpt_name)
#    graph_def = tf.python.graph_util.convert_variables_to_constants(session, session.graph_def, dst_nodes )

#g_2 = tf.Graph()
#with tf.Session(graph=g_2) as sess:
#    tf.train.write_graph(
#    tf.python.graph_util.extract_sub_graph(
#    graph_def, dst_nodes), path, fname, as_text=False)