import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.datasets import mnist
import pickle

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Preprocess the dataset
# Conversion to floats ensures the possibility of getting decimals after division
X_train = X_train.astype('float32')
X_train = X_train / 255.0
X_test = X_test.astype('float32')
X_test = X_test / 255.0
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)
Y_test = to_categorical(Y_test, num_classes = 10)


# Derive the validation set from the training data 
# 80% train, 10% test, 10% validation
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
# A second train_test_split() is called in order to further separate the new X_train, Y_train dataset into train and validation
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)


# The datagen will randomly augment some of the images in the dataset in order to increase the varaibility in the images. 
# This will prevent overfitting.
datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1)

datagen.fit(X_train)


'''
Conv2D - layer versatile for reading image data 
MaxPool2D - down samples the feature map by taking the maximum
Batch Normalization - normalizes the data points by making all feature scales constant
'''
# Define the model
model = Sequential()

model.add(Conv2D(32, kernel_size=5,input_shape=(28, 28, 1), activation = 'relu'))
model.add(Conv2D(32, kernel_size=5, activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=3,activation = 'relu'))
model.add(Conv2D(64, kernel_size=3,activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size=3, activation = 'relu'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(10, activation = "softmax"))

optimizer=Adam(lr=0.001)
model.compile(optimizer=optimizer , loss="categorical_crossentropy", metrics=["accuracy"])


# Train the model and predict
model.fit_generator(datagen.flow(X_train,Y_train, batch_size=128),
                              epochs = 200, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=100)
# Saves the trained model in a pickle file 
pickle.dump(model, open('model.pkl', 'wb'))
