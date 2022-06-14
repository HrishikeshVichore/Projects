# Larger CNN for the MNIST Dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
#print(type(X_train),type(y_train),type(X_test),type(y_test))
print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

"""This time we define a large CNN architecture with additional convolutional, 
max pooling layers and fully connected layers. The network topology can be summarized as follows.

Convolutional layer with 30 feature maps of size 5×5.

Pooling layer taking the max over 2*2 patches.

Convolutional layer with 15 feature maps of size 3×3.

Pooling layer taking the max over 2*2 patches.

Dropout layer with a probability of 20%.

Flatten layer.

Fully connected layer with 128 neurons and rectifier activation.

Fully connected layer with 50 neurons and rectifier activation.

Output layer."""

# define the larger model
def larger_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = larger_model()
for i in range(8):  
    print(model.layers[i].output)
model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs = 10, batch_size = 200, verbose = 2)
scores = model.evaluate(X_test, y_test, verbose = 0)
print("Large CNN Error = ",(100 - scores[1]*100))


"""
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
"""