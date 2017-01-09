import json
import numpy as np


train_dataset_a, train_labels_a = [], []
train = np.array([np.empty([40,40,40])])
for counter in range(2,21):
	with open('/Volumes/EXT WD Elements 1 TB/TermProject/termProject/Russlan data/training_set copy ' + str(counter) + '.json', 'r') as t:
	  	trFile = json.load(t)
	  	train_dataset = np.array(trFile['train_dataset']).astype(np.float32)
	  	train = np.append(train, train_dataset, axis = 0)
	  	train_labels = np.array(trFile['train_labels']).astype(np.float32)
	  	train_labels=train_labels.reshape(train_labels.shape[0],1)
		#label left out
	  	del trFile  #  to help gc free up memory
	  	t.close
	  	print('Counter: ' + str(counter) + ' Training set', train_dataset.shape, train_labels.shape)
print train.shape