# import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# multi perceptron model
# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
# X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# simple cnn model
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# define multiple perceptrons model
def perceptrons_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# define simple cnn model
def cnn_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = cnn_model()
# model = perceptrons_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# model.save('perceptrons_model.h5')
model.save('cnn_model.h5')
print("Save model successfully")

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))


# ================ load own data for testing ================
# The code below loads my own data into the trained model to predict the student number
# we need to make the accuracy to achieve more than 98%

# test student number image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())
image = cv2.imread(args['image'])

GROUND_TRUTH = [5, 6, 7, 8, 9, 8, 7, 7, 2]
GT = np_utils.to_categorical(GROUND_TRUTH)

# Resize the student number picture
image = cv2.resize(image, (252, 28))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imwrite('thresh_student_number.png', thresh)

# vertically split student number into 9 digits
cells = [np.hsplit(thresh, 9)]

# the shape is (1, 9, 28, 28), which stands for 9 numbers, each number consists of 28*28 pixels
x = np.array(cells)

# the shape of flattened matrix is (9, 784), which stands for 9 arrays and length of each array is 784
# student_number = x.reshape(-1, 784).astype(np.float32)
student_number = x.reshape(9, 1, 28, 28).astype(np.float32)

# output the predicted class for my sample
scores = model.predict(student_number, verbose=0)
print(np.argmax(scores, axis=1))

# evaluate the model accuracy
scores = model.evaluate(student_number, GT, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
