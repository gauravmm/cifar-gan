"""
A simple model, containing both generator and discriminator.
"""
from keras import layers, models
from typing import Tuple

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU


#
# GENERATOR
#

# Name of generator
NAME="Simple"
# Size of random seed used by the generator's input tensor:
SEED_DIM = (32,)
# Size of the output. The generator, discriminator and dataset will be required to use this size.
IMAGE_DIM = (32, 32, 3)

def generator(input_size, output_size) -> layers.convolutional._Conv:
    # We only allow the discriminator model to work on CIFAR-sized data.
    assert output_size == (32, 32, 3)
    (img_height, img_width, img_channels) = output_size

    dim = 4
    assert img_height % dim == 0 and img_width % dim == 0, \
        'Generator network must be able to transform `x` into a tensor of shape (img_height, img_width, img_channels).'

    #
    # input dimensions to the first conv layer in the generator
    #
    model = Sequential()

    model.add(Dense(dim * dim * 16, input_shape=input_size))
    model.add(Activation('relu'))
    model.add(layers.Reshape((dim, dim, -1)))
    features = 512
    while dim != img_height:
        dim *= 2
        features //= 2
        
        model.add(Conv2DTranspose(features, (3, 3),
                                  padding = 'same', strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
    # number of feature maps => number of image channels
    model.add(Conv2DTranspose(img_channels, (1, 1), activation='tanh', padding='same'))
    
    return model


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

    model = Sequential()
    
    dim = 4
    model.add(Dense(dim * dim * 16, input_shape=input_size))
    # down sample with strided convolutions until we reach the desired spatial dimension (4 * 4 * features)
    features = 64
    while img_height > dim:
        dim *= 2
        features *= 2
        model.add(Conv2D(features, (3, 3), padding='same', strides=(2, 2)))
        model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(16))
    model.add(LeakyReLU())
    model.add(Dense(1, activation='sigmoid'))

    return model
