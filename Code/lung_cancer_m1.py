
"""
A convolutional neural network to detect lung cancer.

Terminology of terms used is similar to from https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/index.html

@author Fabian Falck
"""

#given data variable: lung_train and lung_test
#MIND: dataset has to be a tensorflow object, since nextBatch method is called on it!!!!!

#TBD data import
#TBD batching, check: https://www.tensorflow.org/versions/r0.12/how_tos/reading_data/index.html#batching



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

W_fc1 = weight_variable([10 * 10 * 10 * 60, 1000])  #does this layer work like that?
b_fc1 = bias_variable([1000])

h_pool2_flat = tf.reshape(h_pool2, [-1, 10*10*10*60]) #flattening the vector again!!!
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout

keep_prob = tf.placeholder(tf.float32) #defined as a placeholder so that it can be turned on and off during training and testing
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Final softmax layer to generate wished output

W_fc2 = weight_variable([1000, 2]) #output channel: 2
b_fc2 = bias_variable([2])
y_est = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 #estimate of y_true!

"""
"""

#running training

sess = tf.Session()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_est, y_true))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #used below
correct_prediction = tf.equal(tf.argmax(y_est,1), tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(1):
  batch = lung_train.next_batch(50) #what is the next_batch() method exactly doing? taking values from validation set? or training set?
  if i%1 == 0:
      #evaluating the set with validation = training set
    train_accuracy = accuracy.eval(session=sess, feed_dict={
        x: batch[0], y_true: batch[1], keep_prob: 1.0}) #during testing no dropout
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(session=sess, feed_dict={x: batch[0], y_true: batch[1], keep_prob: 0.5}) #during training 0.5 dropout at the certain layer

print("test accuracy %g"%accuracy.eval(session = sess, feed_dict={
    x: lung_train.images, y_true: lung_test.labels, keep_prob: 1.0}))








  #Ideas:

#Check out each type of nodule (see 6 types) performs and improve the algorithm based on this knowledge

#UML Diagramme of the data structure