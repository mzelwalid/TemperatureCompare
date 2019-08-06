import pprint
import itertools
import numpy
import tensorflow as tf
import tensorflow.keras as k
from random import sample

#Class
class ProteinMatrixConcatInput(k.utils.Sequence):
	def __init__(self, x, batch_size, pad_to):
		self.x = x
		self.perms = tuple(itertools.permutations(x.keys(), 2))
		self.batch_size = batch_size
		self.sample_size = len(self.perms)
		self.pad_to = pad_to
	
	def __len__(self):
		return int(numpy.ceil(self.sample_size / float(self.batch_size)))
	
	def __getitem__(self, idx):
		#Step 1: extract appropriate permutations from self.perms
		perms_batch = self.perms[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_x = []
		batch_y = []

		for i, pair in enumerate(perms_batch):
			leftOGT = pair[0].split('_')[2]
			rightOGT = pair[1].split('_')[2]
			rightGreater = int(leftOGT < rightOGT)
			combination = numpy.vstack((self.x[pair[0]], self.x[pair[1]]))[:,2:]
			batch_y.append(rightGreater)
			# pad batch_x here
			pad = numpy.zeros((self.pad_to - combination.shape[0], 26))
			combination = numpy.vstack((combination, pad))
			batch_x.append(numpy.expand_dims(combination, axis = 2))

		batch_x = numpy.stack(batch_x, axis = 0)
		batch_y = numpy.asarray(batch_y)
		return (batch_x, batch_y);


dictionary = {}
filename = "data/Ku40SecondaryStructures_with_class.txt"

with open(filename) as data_file:
	protKey = ''
	for line in data_file.readlines():
		line = line.strip()
		if line.startswith('>'):
			dictionary[line[1:]] = []
			protKey = line[1:]
		else:
			dictionary[protKey].append(line.split("\t"))

perms = itertools.permutations(dictionary.keys(), 2)


#convert lists in dictionary to numpy arrays
numpy_dict = {}
for key in dictionary:
	numpy_dict[key] = numpy.asarray(dictionary[key])

# find longest concatenation in perms
max_length = 0
for ele in perms:
	max_length = max(numpy_dict[ele[0]].shape[0] + numpy_dict[ele[1]].shape[0], max_length)

all_keys = list(numpy_dict.keys())
num_of_val_keys = int(numpy.ceil(len(all_keys) * 0.1))

val_keys = sample(all_keys, num_of_val_keys)

train_numpy_dict = {key: numpy_dict[key] for key in all_keys if key not in val_keys}
val_numpy_dict = {key: numpy_dict[key] for key in all_keys if key in val_keys}


m = k.Sequential()

#Add convolutional layer
m.add(k.layers.Conv2D(
	filters = 24, 
	kernel_size = (8,26), 
	strides = (6,26), 
	input_shape = (max_length, 26, 1)))
m.add(k.layers.Flatten())

#Add other layers
m.add(k.layers.Dense(units = 8))
m.add(k.layers.Dense(
	units = 1,
	activation = k.activations.sigmoid))

#Compile the model
m.compile(
	loss = 'binary_crossentropy',
	optimizer = k.optimizers.SGD(lr = 0.001),
	metrics = ['accuracy'])

print(m.summary())

history = m.fit_generator(
	generator = ProteinMatrixConcatInput(train_numpy_dict, 64, max_length),
	epochs = 5,
	validation_data = ProteinMatrixConcatInput(val_numpy_dict, 64, max_length),
	workers = 2)

print(history)
















