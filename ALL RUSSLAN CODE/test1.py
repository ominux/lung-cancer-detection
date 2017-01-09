# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import numpy as np
import tensorflow as tf
from six.moves import range
import json

training_dir = '/Volumes/EXT WD Elements 1 TB/TermProject/termProject/Russlan data/training_set.json'
validation_dir = '/Volumes/EXT WD Elements 1 TB/TermProject/termProject/Russlan data/validation_set.json'

with open(training_dir, 'r') as t:
  trFile = json.load(t)
  train_dataset = np.array(trFile['train_dataset']).astype(np.float32)
  train_labels = np.array(trFile['train_labels']).astype(np.float32)
  train_labels=train_labels.reshape(train_labels.shape[0],1)
  del trFile  #  to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)