# Import the essential libraries
import os
import csv
import cv2
import matplotlib.image as npimg
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
# from keras.layers.pooling import MaxPooling2D
from keras import regularizers
# from sklearn.preprocessing import LabelBinarizer
import math

samples = []

# Read the CSV file from data and put into variable
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        samples.append(line)

# Split the data into train and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Define the generator function to take the images in bunches, this is to efficient use of the available memory
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples) # shuffle the images array to improve the learning
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Read the center image and append
                name = '../data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = npimg.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                # Flip the center image
                images.append(np.fliplr(center_image))
                angles.append(center_angle * -1)
                # Set correction value and use it for correct the angle of left and right images
                correction = 0.2 
                # Read left image and append
                name = '../data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = npimg.imread(name)
                left_angle = center_angle + correction #float(batch_sample[3])
                images.append(left_image)
                angles.append(left_angle)
                
                # Flip the left image
                images.append(np.fliplr(left_image))
                angles.append(left_angle * -1)
                
                # Open right image and append
                name = '../data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = npimg.imread(name)
                right_angle = center_angle - correction # float(batch_sample[3])
                images.append(right_image)
                angles.append(right_angle)
                
                # Flip the right image
                images.append(np.fliplr(right_image))
                angles.append(right_angle * -1)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# geting the batch data using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

ch, row, col = 3, 160, 320  # Original image format
# ch, row, col = 3, 80, 320  # Trimmed image format
# height 160     center_2016_12_01_13_46_38_543.jpg
# width 320

# Intentiate the model
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.0,
        input_shape=(row, col, ch) ))
# Crop the images and ignore the unnecessary data eg. sky at the upper and car body near by camera
model.add(Cropping2D(cropping=((50,20), (0,0)) )) # trim image to only see section with road
# Covolutional layer with 24 channels, 5 X 5 kernel, strides 2 X 2
model.add(Conv2D(24, (5, 5), strides=(2,2), padding='valid'))
model.add(Activation('elu'))
model.add(Conv2D(36, (5, 5), strides=(2,2), padding='valid'))
model.add(Activation('elu'))
model.add(Conv2D(48, (5, 5), strides=(2,2), padding='valid'))
model.add(Activation('elu'))
model.add(Conv2D(64, (3, 3), strides=(1,1), padding='valid'))
model.add(Activation('elu'))
model.add(Conv2D(64, (3, 3), strides=(1,1), padding='valid'))
model.add(Activation('elu'))
model.add(Flatten())
model.add(Dense(100, kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('elu'))
model.add(Dense(50, kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('elu'))
model.add(Dense(10, kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('elu'))
model.add(Dense(1))
# compile and train the model using the generator
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/128), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/128), epochs=5)
# Saving the model
model.save('behavioral_cloning.h5')
