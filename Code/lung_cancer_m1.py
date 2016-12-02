
"""
A convolutional neural network to detect lung cancer.

Terminology of terms used is similar to from https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/index.html

@author Fabian Falck
"""


import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 40, 40, 40])
y_true = tf.placeholder(tf.float32, [None, 2])

#function directly copied from https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/index.html
#avoids gradient of 0
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#function directly copied from https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/index.html
#avoids "dead neuron"
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def max_pool_2x2x2(x): #Mind: this max-pooling reduces each dimension by 2
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 2, 1],
                        strides=[1, 2, 2, 2, 1], padding='SAME')

#Layer 1

W_conv1 = weight_variable([5, 5, 5, 1, 30])
b_conv1 = bias_variable([30])

h_conv1 = tf.nn.relu(conv3d(x, W_conv1) + b_conv1) #MIND: x vector has to be already reshaped, otherwise use sth like x_image = tf.reshape(x, [-1,28,28,1])
h_pool1 = max_pool_2x2x2(h_conv1) #mind: max pooling causes dimension reduction! --> current: 20x20x20

#Layer 2

W_conv2 = weight_variable([5, 5, 5, 30, 60])
b_conv2 = bias_variable([60])

h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2x2(h_conv2) #current dimension: 10x10x10

#Fully connected layer

W_fc1 = weight_variable([10 * 10 * 60, 1000])
b_fc1 = bias_variable([1000])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


  #Ideas:

#Check out each type of nodule (see 6 types) performs and improve the algorithm based on this knowledge

#UML Diagramme of the data structure