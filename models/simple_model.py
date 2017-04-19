"""
A simple model, containing both generator and discriminator.
"""
from typing import Tuple

from keras import layers, models
from keras.layers import (Activation, Conv2D, Conv2DTranspose, Dense, Dropout,
                          Flatten, Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Lambda
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import tensorflow as tf

#
# GENERATOR
#

# Name of generator
NAME="Simple"
# Size of random seed used by the generator's input tensor:
SEED_DIM = (32,)
# Size of the output. The generator, discriminator and dataset will be required to use this size.
IMAGE_DIM = (32, 32, 3)
#leaky relu coefficient
alpha = 0.3

#hand-made normalization
def normalization(x):
    x -= K.mean(x, axis=0, keepdims=True)
    x /= K.std(x, axis=0, keepdims=True)
    return x


def generator(input_size, output_size) -> layers.convolutional._Conv:
    # We only allow the discriminator model to work on CIFAR-sized data.
    assert output_size == (32, 32, 3)
    
    x = layers.Input(shape=input_size, name="input_gen_seed")
    y = x
    lys = [
        Dense(256),
        #Lambda(normalization),
        #BatchNormalization(),
        LeakyReLU(0),
        Reshape((4, 4, -1)),
        Conv2DTranspose(256, (3, 3), padding = 'same', strides=(2, 2)),
        #Lambda(normalization),
        #BatchNormalization(),
        LeakyReLU(0),
        #Dropout(0.5),
        Conv2DTranspose(128, (3, 3), padding = 'same', strides=(2, 2)),
        #Lambda(normalization),
        #BatchNormalization(),
        LeakyReLU(0),
        #Dropout(0.5),
        Conv2DTranspose(64, (3, 3), padding = 'same', strides=(2, 2)),
        #Lambda(normalization),
        #BatchNormalization(),
        LeakyReLU(0),
        #Dropout(0.5),
        Conv2DTranspose(3, (1, 1), activation='tanh', padding='same')
    ]

    for l in lys:
        y = l(y)

    return models.Model(inputs=x, outputs=y, name="model_generator")


#
# DISCRIMINATOR
#

# Name of discriminator
# Each generator/discriminator needs a name if they are in different files
# NAME="Simple"

def discriminator(input_size):
    # We only allow the discriminator model to work on CIFAR-sized data.
    assert input_size == (32, 32, 3)
    (img_height, img_width, img_channels) = input_size

    x = layers.Input(shape=input_size, name='input_dis')
    y = x
    lys = [
    	Dense(256),
        #Dropout(0.3),
    	Conv2D(128, (3, 3), padding='same', strides=(2, 2)),
        #Dropout(0.5),
        #Lambda(normalization),
        #BatchNormalization(),
    	LeakyReLU(alpha),
       	Conv2D(256, (3, 3), padding='same', strides=(2, 2)),
        #Dropout(0.5),
        #Lambda(normalization),
        #BatchNormalization(),
    	LeakyReLU(alpha),
    	Conv2D(512, (3, 3), padding='same', strides=(2, 2)),
        #Dropout(0.5),
        #Lambda(normalization),
        #BatchNormalization(),
    	LeakyReLU(alpha),
    	Flatten(),
        Dense(16),
        #Dropout(0.3),
        #Lambda(normalization),
        #BatchNormalization(),
        LeakyReLU(alpha),
        Dense(1, activation='sigmoid', name='discriminator')
    ]

    for l in lys:
        y = l(y)
    
    return models.Model(inputs=x, outputs=y, name="model_discriminator")
