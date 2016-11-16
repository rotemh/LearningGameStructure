from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np

NUM_EXAMPLES = 10000
NUM_VALIDATION_POINTS = 10

model = Sequential()

model.add(Dense(output_dim=5000,input_dim=3))
model.add(Activation("relu"))
model.add(Dense(output_dim=5000))
model.add(Activation("relu"))
#model.add(Dense(output_dim=2))
model.add(Dense(output_dim=1))
model.add(Activation("linear"))
model.compile(optimizer='rmsprop',loss="mse",metrics=["mean_squared_error"])

data = np.random.random((NUM_EXAMPLES,3))
f_a = lambda x,y,z: np.multiply(x,y) + 3*z
f_b = lambda x,y,z: np.multiply(x,y) - 3*z
labels_a = f_a(data[:,0],data[:,1],data[:,2])
labels_b = f_b(data[:,0],data[:,1],data[:,2])
#labels = np.asarray([labels_a,labels_b]).transpose()
labels = labels_a

model.fit(data,labels,nb_epoch=10,batch_size=32)

validation_data = np.random.random((NUM_VALIDATION_POINTS,3))
validation_labels_a = f_a(data[:,0],data[:,1],data[:,2])
validation_labels_b = f_b(data[:,0],data[:,1],data[:,2])
predicted_labels = model.predict(validation_data,verbose=1)
for i in xrange(NUM_VALIDATION_POINTS):
	print "Example 1 - "
	print "a: ", validation_data[i,0], \
		", b: ", validation_data[i,1], \
		", c: ", validation_data[i,2]
	# print "correct result: ", validation_labels_a[i], ", ", validation_labels_b[i]
	# print "true result: ", predicted_labels[i,0], ", ", predicted_labels[i,1]
	print "correct result: ", validation_labels_a[i]#, ", ", validation_labels_b[i]
	print "true result: ", predicted_labels[i]#, ", ", predicted_labels[i,1]

