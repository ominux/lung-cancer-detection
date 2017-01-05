import json
import numpy as np


for counter in range(2,21):
	with open('/Volumes/EXT WD Elements 1 TB/TermProject/termProject/Russlan data/training_set copy ' + str(counter) + '.json', 'r') as t:
	  	trFile = json.load(t)
	  	train_dataset = np.array(trFile['train_dataset']).astype(np.float32)
	  	train_labels = np.array(trFile['train_labels']).astype(np.float32)
	  	train_labels=train_labels.reshape(train_labels.shape[0],1)
	  	del trFile  #  to help gc free up memory
	  	t.close
	  	print('Counter: ' + str(counter) + ' Training set', train_dataset.shape, train_labels.shape)