import pprint
import itertools
import numpy
import tensorflow as tf
import tensorflow.keras as k

#Class
class ProteinMatrixConcatInput(k.utils.Sequence):
	def __init__(self, x, batch_size):
		self.x = x
		self.perms = tuple(itertools.permutations(x.keys(), 2))
		self.batch_size = batch_size
		self.sample_size = len(self.perms)
	
	def __len__(self):
		return int(numpy.ceil(self.sample_size / float(self.batch_size)))
	
	def __getitem__(self, idx):
		#Step 1: extract appropriate permutations from self.perms
		perms_batch
	


dictionary = {}
filename = "Ku40SecondaryStructures_with_class.txt"

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
perms_slice = list(itertools.islice(perms, 500))


#convert lists in dictionary to numpy arrays
numpy_dict = {}
for key in dictionary:
	numpy_dict[key] = numpy.asarray(dictionary[key])
	
x = []
y = []

# find longest concatenation in perms
max_length = 0
for ele in perms_slice:
	max_length = max(numpy_dict[ele[0]].shape[0] + numpy_dict[ele[1]].shape[0], max_length)

for i, pair in enumerate(perms_slice):
	leftOGT = pair[0].split('_')[2]
	rightOGT = pair[1].split('_')[2]
	rightGreater = int(leftOGT < rightOGT)
	combination = numpy.vstack((numpy_dict[pair[0]], numpy_dict[pair[1]]))[:,2:]
	y.append(rightGreater)
	# pad x here
	pad = numpy.zeros((max_length - combination.shape[0], 26))
	combination = numpy.vstack((combination, pad))
	x.append(numpy.expand_dims(combination, axis = 2))

x = numpy.stack(x, axis = 0)


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

history = m.fit(
	x = x,
	y = y,
	batch_size = 10,
	epochs = 40,
	validation_split = 0.2)

print(history)
















