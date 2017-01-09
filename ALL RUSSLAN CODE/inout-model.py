# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import numpy as np
import tensorflow as tf
from six.moves import range
import json
import scipy.io as sio


resume = False

training_dir = '/Volumes/EXT WD Elements 1 TB/TermProject/termProject/Russlan data/training_set.json'
validation_dir = '/Volumes/EXT WD Elements 1 TB/TermProject/termProject/Russlan data/validation_set.json'

with open(training_dir, 'r') as t:
  trFile = json.load(t)
  train_dataset = np.array(trFile['train_dataset']).astype(np.float32)
  train_labels = np.array(trFile['train_labels']).astype(np.float32)
  train_labels=train_labels.reshape(train_labels.shape[0],1)
  del trFile  #  to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)

#ATTENTION: SAME AS TRAINING DATASET
"""
with open(validation_dir, 'r') as v:
  trFile = json.load(v)
  valid_dataset = np.array(trFile['test_dataset']).astype(np.float32)
  valid_labels = np.array(trFile['test_labels']).astype(np.float32)
  valid_labels=valid_labels.reshape(valid_labels.shape[0],1)
  del trFile  #  to help gc free up memory
  print('Validation set', valid_dataset.shape, valid_labels.shape)
"""

valid_dataset = train_dataset
valid_labels = train_labels

print 'Input loaded.'

image_size = 40
num_labels = 1
num_channels = 40 # 3D


def accuracy(predictions, labels):
 # print("sum of predictions = ", np.sum(predictions))
  #print("sum of labels = ", np.sum(labels))
  return (100.0 * np.sum(predictions == labels)/ predictions.shape[0])
  
batch_size = 600
patch_size = 5
depth = 80
num_hidden = 300

def savingPredictions(predictions, labels): #add: 'IDs':IDs,
  outPath = '/Users/FabianFalck/ML/lung-cancer-detection/Results/predict_results.mat'
  sio.savemat(outPath, {'predictions':predictions, 'labels': labels}) #add: 'IDs':IDs,

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

  print 'model loaded'

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
  
  saver = tf.train.Saver() #SAVING

num_steps = 300



with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  prev_loss=0
  print 'starting the steps'
  for step in range(num_steps):
    " the following offset is used if stochastic grad descent is used, but as I am using whole training data offset is not necessary"
    # offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    #batch_labels = train_labels[offset:(offset + batch_size)]
    batch_data = train_dataset
    batch_labels = train_labels                      
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels:batch_labels}
    _, l, predictions,sigm = session.run(
      [optimizer, loss, train_prediction,xent], feed_dict=feed_dict)
    if (step %  10==0):
      print 'Current step: ',step
      print('Minibatch loss at step %d: %f' % (step, l))
      #print('Sigmoid function:' ,  np.sum(sigm))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      predictions = valid_prediction.eval() #gives back 0 or 1
      print('Validation accuracy: %.1f%%' % accuracy(predictions, valid_labels))
      savingPredictions(predictions,valid_labels) #add: IDs

      modelSavePath = '/Users/FabianFalck/ML/lung-cancer-detection/Results/my-model'
      saver.save(session, 'my-model', global_step=step) #SAVING

    #being super optimistic in adding the following case:
    #if np.abs(prev_loss - loss) < 0.00001:
            #break
    # Keep track of the training cost
    prev_loss=loss




    #helpful links:
    #saving and laoding: http://stackoverflow.com/questions/36281129/no-variable-to-save-error-in-tensorflow
        #https://www.tensorflow.org/api_docs/python/state_ops/saving_and_restoring_variables#Saver