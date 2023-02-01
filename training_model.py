from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, SpatialDropout2D, Flatten, Dropout, Dense
import keras.utils as image
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from data_preprocessing import data, target
from tensorflow.keras.callbacks import ModelCheckpoint



model = Sequential()
model.add(Conv2D(200, (3, 3), activation='relu', input_shape=data.shape[1:])) #Filters, #Kernal size #Relu
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu')) #Filters, #Kernal size #Relu
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu')) #Filters, #Kernal size #Relu
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50, activation="relu"))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1)

checkpoint = ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
history = model.fit(train_data, train_target, epochs=10, callbacks=[checkpoint], validation_split=0.2)
print(model.evaluate(test_data, test_target ))