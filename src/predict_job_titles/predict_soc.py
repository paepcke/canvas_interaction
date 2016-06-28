from pymongo import MongoClient
import csv

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Highway
from keras.optimizers import SGD

def glove2dict(src_filename):
    """GloVe Reader.
    
    Parameters
    ----------
    src_filename : str
        Full path to the GloVe file to be processed.
    Returns
    -------
    dict
        Mapping words to their GloVe vectors.
    
    """
    reader = csv.reader(open(src_filename), delimiter=' ', quoting=csv.QUOTE_NONE)    
    return {line[0]: np.array(list(map(float, line[1: ]))) for line in reader}

def major2index(major):
	return (int(major) - 11) / 2

def minor2index(minor):
	indices = []
	for i in range(len(minor)):
		digit = int(minor[i])
		indices.append((i * 10 + digit) + 23)
	return indices

def specific2index(specific):
	indices = []
	for i in range(len(specific)):
		digit = int(specific[i])
		indices.append((i * 10 + digit) + 63)
	return indices

input_size = 50
output_size = 83

glove_dict = glove2dict('glove.6B/glove.6B.%dd.txt' % input_size)

# get code from mongodb database and create x and y
client = MongoClient()
db = client.soc
soc = db.soc_4_million
count = 0
X = []
y = []
for s in soc.find().batch_size(500):
	if count % 1000 == 0:
		print "got %d things" % count
	# print s['filtered_title']
	# print s['job_description']
	# print s['skills']

	vec = np.zeros(input_size)
	for w in s['filtered_title'].split():
		w = str(w).strip()
		if len(w) < 1:
			continue
		print w
		if w in glove_dict.keys():
			vec += glove_dict[w]
		else:
			print "couldn't find word", repr(w)

	X.append(vec)
	onet_code, specific = s['most_probable_onet_code'][0].split('.')
	major, minor = onet_code.split('-')

	y_vec = np.zeros(output_size)
	y_vec[major2index(major)] = 1
	y_vec[minor2index(minor)] = 1
	y_vec[specific2index(specific)] = 1
	y.append(y_vec)

	count += 1

X = np.matrix(X)
y = np.matrix(y)

print "created matrices"

#split x and y into train and test sets
assert(len(X) == len(y))

num_samples = len(X)

num_test = num_samples * 0.1
test_rows = np.random.randint(num_samples, size=num_test)

X_test = X[test_rows,:]
y_test = y[test_rows,:]

assert(len(X_test) == len(y_test))

train_rows = list(set(range(num_samples)) - set(test_rows))

X_train = X[train_rows,:]
y_train = y[train_rows,:]

assert(len(X_train) == len(y_train))


model = Sequential()

# # stacking layers is easy
# # model.add(Dense(output_dim=64, input_dim=100))
# # model.add(Activation("relu"))
# # model.add(Dense(output_dim=10))
# # model.add(Activation("softmax"))

# # add highway layer
# '''
# Parameters:
# 	init: name of initiliazation function for weights of the layer
# 		(only relevant if you don't pass a weights argument)
# 	transform_bias: value for bias to take on initially (default -2)
# 	activation: name of activation function to use, or alternatively, elementwise Theano function
# 		If you don't specify anything, no activiation is applied (i.e. "linear" activation: a(x) = x)
# 	weights: list of numpy arrays to set as initial weights
# 		Should have 2 elements of shape (input_dim, output_dim) and (output_dim,) for weights and biases respectively
# 	W_regularizer: instance of WeightRegularizer (eg. L1 or L2 regularization), applied to the main weights mtarix
# 	b_regularizer: instance of WeightRegularizer, applied to the bias
# 	activity_regularizer: instance of ActivityRegularizer, applied to the network output
# 	W_constraint: instance of the constraints module (eg. maxnorm, nonneg), applied to main weights matrix
# 	b_constraint: instance of constraints module, applied to the bias
# 	bias: whether to include a bias (i.e. make the layer affine rather than linear)
# 	input_dim: dimensionality of the input (integer) - required when using this layer as the first layer in the model.
# '''
# #initial Highway layer
model.add(Highway(init="glorot_uniform", activation="tanh", input_dim = input_size))

for i in range(50):
	model.add(Highway(init="glorot_uniform", activation="tanh"))

model.add(Dense(output_dim=output_size))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# # similar to above but allows you to further configure optimizer
# #model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

# # iterate on training data in batches
model.fit(X_train, y_train, nb_epoch=5, batch_size=32)

# # can feed batches to model manually
# #model.train_on_batch(X_batch, Y_batch)

# # evaluate performance in one line:
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=32)

print loss_and_metrics

# # generate predictions on new data
# classes = model.predict_classes(X_test, batch_size=32)
# proba = model.predict_proba(X_test, batch_size=32)
