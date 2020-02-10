# import cv2
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras import backend as K
# import time 
# import os

img_width, img_height = 20, 20

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

model = Sequential()

model.add(ZeroPadding2D(padding=(2,2),input_shape=(20,20,1)))

model.add(Conv2D(16,(3,3),activation='relu')) 

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3),activation='relu')) 

model.add(Flatten())

model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

dirTrain = 'Training-Process/Training-Database/'
dirValidation = 'Training-Process/Validating-Database/'
dirTrainTertutup = 'Training-Process/Training-Database/0-Tertutup/'
dirTrainTerbuka = 'Training-Process/Training-Database/1-Terbuka/'
dirValidationTertutup = 'Training-Process/Validating-Database/0-Tertutup/'
dirValidationTerbuka = 'Training-Process/Validating-Database/1-Terbuka/'

nb_train_samples = 400*2
nb_validation_samples = 100*2
epochs = 5
batch_size = 32

train_datagen = ImageDataGenerator(
        rescale=1./255,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True
        )

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        dirTrain,
        target_size=(20, 20),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        dirValidation,
        target_size=(20, 20),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary')

# model.fit()
# model.fit(dirTrainTertutup,dirTrainTerbuka, batch_size=batch_size, nb_epoch=epochs,
#     verbose=1, validation_data=(dirValidationTertutup, dirValidationTertutup))

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# model.save_weights('BismillahFirst-5epochs-W.h5')
model.save('BismillahFirst-5epochs.h5')