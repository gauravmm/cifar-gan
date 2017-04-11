"""
A simple model, containing both generator and discriminator.
"""
from keras import layers, models
from typing import Tuple

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Conv2DTranspose
from keras.layers.merge import Concatenate
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
#leaky relu coefficient
alpha = 0.3

def generator(inp, inp_label, output_size) -> Tuple[layers.convolutional._Conv, layers.convolutional._Conv]:
    # We only allow the discriminator model to work on CIFAR-sized data.
    assert output_size == (32, 32, 3)
    (img_height, img_width, img_channels) = output_size

    assert img_height % 4 == 0 and img_width % 4 == 0, \
        'Generator network must be able to transform `x` into a tensor of shape (img_height, img_width, img_channels).'

    x = Concatenate()(Flatten()(inp), Flatten()(inp_label))

    layers = [
        Dense(256)
        LeakyReLU(alpha)
        layers.Reshape((4, 4, -1)),
        Conv2DTranspose(256, (3, 3), padding = 'same', strides=(2, 2)),
        LeakyReLU(alpha),
        Conv2DTranspose(128, (3, 3), padding = 'same', strides=(2, 2)),
        LeakyReLU(alpha),
        Conv2DTranspose(64, (3, 3), padding = 'same', strides=(2, 2)),
        LeakyReLU(alpha),
        Conv2DTranspose(img_channels, (1, 1), activation='tanh', padding='same')
    ]

    for l in layers:
        x = l(x)

    model = Model(inputs=[inp, inp_label], outputs=x)

    return model


#
# DISCRIMINATOR
#

# Name of discriminator
# Each generator/discriminator needs a name if they are in different files
# NAME="Simple"

def discriminator(inp, num_classes):
    # We only allow the discriminator model to work on CIFAR-sized data.
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

    y1 = Dense(16)(x)
    y1 = LeakyReLU(alpha)(y1)
    y1 = Dense(1, activation='sigmoid')(y1)

    y2 = Dense(num_classes)(x)
    y2 = LeakyReLU(alpha)(y2)
    y2 = Dense(1, activation='sigmoid')(y2)

    model_fake  = Model(inputs=[inp], outputs=[y1])
    model_class = Model(inputs=[inp], outputs=[y2])

    return (model_fake, model_class)

