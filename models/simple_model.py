"""
A simple model, containing both generator and discriminator.
"""
import tensorflow as tf
from keras import backend as K
from keras import layers, models
from keras.layers import (Activation, Conv2D, Conv2DTranspose, Dense, Dropout,
                          Flatten, Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate
from keras.models import Model, Sequential
from typing import Tuple

#
# GENERATOR
#

# Name of generator
NAME="Simple"
#leaky relu coefficient
alpha = 0.3

def normalization(x):
    x -= K.mean(x, axis=0, keepdims=True)
    x /= K.std(x, axis=0, keepdims=True)
    return x

def generator(inp, inp_label, output_size) -> Tuple[layers.convolutional._Conv, layers.convolutional._Conv]:
    # We only allow the discriminator model to work on CIFAR-sized data.
    assert output_size == (32, 32, 3)
    (img_height, img_width, img_channels) = output_size

    assert img_height % 4 == 0 and img_width % 4 == 0, \
        'Generator network must be able to transform `x` into a tensor of shape (img_height, img_width, img_channels).'

    x = Concatenate()([inp, inp_label])

    layers = [
        Dense(256),
        LeakyReLU(0),
        Reshape((4, 4, -1)),
        Conv2DTranspose(256, (3, 3), padding = 'same', strides=(2, 2)),
        LeakyReLU(0),
        Conv2DTranspose(128, (3, 3), padding = 'same', strides=(2, 2)),
        LeakyReLU(0),
        Conv2DTranspose(64, (3, 3), padding = 'same', strides=(2, 2)),
        #Lambda(normalization),
        LeakyReLU(0),
        Conv2DTranspose(3, (1, 1), activation='tanh', padding='same')
    ]

    for l in layers:
        x = l(x)

    return [x]


#
# DISCRIMINATOR
#

# Name of discriminator
# Each generator/discriminator needs a name if they are in different files
# NAME="Simple"

def discriminator(inp, num_classes):
    x = inp

    layers = [
    	Dense(256),
    	Conv2D(128, (3, 3), padding='same', strides=(2, 2)),
    	LeakyReLU(alpha),
       	Conv2D(256, (3, 3), padding='same', strides=(2, 2)),
    	LeakyReLU(alpha),
    	Conv2D(512, (3, 3), padding='same', strides=(2, 2)),
    	LeakyReLU(alpha),
    	Flatten()
    ]

    for l in layers:
        x = l(x)

    # The name parameters here are crucial!
    # The order of definition and inclusion in output is crucial as well! You must define y1 before y2, and also include
    # them in output in the order.
    y1 = Dense(16)(x)
    y1 = LeakyReLU(alpha)(y1)
    y1 = Dense(1, activation='sigmoid', name='discriminator')(y1)

    y2 = Dense(num_classes, activation='sigmoid', name='classifier')(x)

    return [y1, y2]
