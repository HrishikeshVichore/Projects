# Plot ad hoc mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt,numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

"""
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
"""
"""
By fixing the seed, you ensure that the ‘random’ numbers generated in your ML algorithm are exactly
the same, every time it is ran. 
This means your experiments/results will be exactly reproducible by yourself and others.
"""
# fix random seed for reproducibility
seed = 8
numpy.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape,X_test.shape)
# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
"""reduce our memory requirements by forcing the precision of the pixel values to be 32 bit"""
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
"""Converting 3-D matrix to a Vector of pixels"""
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
exit()
"""The pixel values are gray scale between 0 and 255. 
It is almost always a good idea to perform some scaling of input values when using neural network models.
Because the scale is well known and well behaved, we can very quickly normalize the pixel values 
to the range 0 and 1 by dividing each value by the maximum of 255."""

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
""" Transform vector of a class integers to a binary matrix  why?"""
print(y_train.shape,y_test.shape)
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print(y_train.shape,y_test.shape,num_classes)


"""Start of development of neural network"""

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    #The Sequential model is a linear stack of layers.
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
# Fit the model
print('X_train = ',X_train.shape,'X_test = ',X_test.shape,
          'Y_train = ',y_train.shape,'Y_test = ',y_test.shape)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=2)
print(scores)
#print("Baseline Error: %.2f%%" % (100-scores[1]*100))