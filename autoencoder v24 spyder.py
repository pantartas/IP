# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:10:40 2022

@author: erikv
"""

#import necessary libraries
import os
import IPython
import librosa as lr
import numpy as np
from datetime import datetime
from packaging import version
import tensorboard
import keras
from keras import layers
from keras import regularizers
import tensorflow as tf
import matplotlib.pyplot as plt
from librosa import display
from tensorflow.keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Flatten,Dense,Activation,Conv2DTranspose
import time
from sklearn import preprocessing

#LOG_DIR = f'{int(time.time())}'

path = 'C:/Users/erikv/Desktop/IP/samples/' #path where the samples are stored'''
dirs = os.listdir(path) #open the directory using os library'''

processed = np.load('processed.npy')[0:100]
xtrain = processed
test = np.array([processed[35]])

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) #check if keras is using GPU optimization
if test[0,20,20,0] == xtrain[35,20,20,0]: #if these two numbers are the same the files are coincident and correct loading
    print ('--------------loading correct--------------')
else:
    print('--------------loading failed. check coordinates--------------') 
    
#the autoencoder is composed of 4 encoding layers and 4 decoding layers

encoder_input = keras.Input(shape=(256, 256, 2), name="original audio")
x = layers.BatchNormalization()(encoder_input)
x = layers.Conv2D(4, 3, activation="relu",padding='same')(x)
x = layers.Conv2D(4, 3, activation="relu",padding='same',strides=(2,2))(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(8, 3, activation="relu",padding='same')(x)
x = layers.Conv2D(8, 3, activation="relu",padding='same',strides=(2,2))(x)
x = layers.Conv2D(16, 3, activation="relu",padding='same')(x)
x = layers.Conv2D(16, 3, activation="relu",padding='same',strides=(2,2))(x)
x = layers.Conv2D(16, 3, activation="relu",padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(32, 3, activation="relu",padding='same',strides=(2,2))(x)
x = layers.Conv2D(32, 3, activation=None,padding='same')(x)
encoder_output = layers.Flatten()(x) #the file is reduced to 8192

encoder = keras.Model(encoder_input, encoder_output, name="encoder")

decoder_input = keras.Input(shape=(8192), name="encoded audio")
x = layers.Reshape((16,16,32))(decoder_input)
x = layers.Conv2DTranspose(32, 3, activation="relu",padding='same')(x)
x = layers.Conv2DTranspose(32, 3, activation="relu",padding='same',strides=(2,2))(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(16, 3, activation="relu",padding='same')(x)
x = layers.Conv2DTranspose(16, 3, activation="relu",padding='same',strides=(2,2))(x)
x = layers.Conv2DTranspose(8, 3, activation="relu",padding='same')(x)
x = layers.Conv2DTranspose(8, 3, activation="relu",padding='same',strides=(2,2))(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(4, 3, activation="relu",padding='same')(x)
x = layers.Conv2DTranspose(4, 3, activation="relu",padding='same',strides=(2,2))(x)
decoder_output = layers.Conv2DTranspose(2, 3, activation=None,padding='same')(x) #this last layer does not have an activation method

decoder = keras.Model(decoder_input, decoder_output, name="decoder")

autoencoder_input = keras.Input(shape=(256, 256, 2), name="input")
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input, decoded_img, name="autoencoder")
autoencoder.summary()

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)    
    
autoencoder.compile(optimizer='adam', 
                  loss='mean_squared_error',
                  metrics=["mean_squared_error"]) #the autoencoder is optimizwed using adam and the loss 
                                                              #and metrics are mean logarithmic squared error
autoencoder.fit(xtrain,xtrain,
                epochs=10,
                batch_size=5,
                shuffle=True,
                validation_split=0.2,
                callbacks = [tensorboard_callback]) #it is trained using a 20% validations. So from the 100 files batch, 80 are for 
                                      #training and 20 are for validation purposes   
 
    
 
    
    