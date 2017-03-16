"""
A simple model, containing both generator and discriminator.
"""
from keras import layers, models
from typing import Tuple

# TODO: Update calls to Keras 2 API to remove warnings

#
# Params
#

# dimensions
img_height = 112
img_width = 112
img_channels = 3

# shared network params
kernel_size = (3, 3)
conv_layer_keyword_args = {'border_mode': 'same', 'subsample': (2, 2)}

# training params
nb_steps = 10000
batch_size = 128
k_d = 1  # number of discriminator network updates per step
k_g = 2  # number of generative network updates per step
log_interval = 100  # interval (in steps) at which to log loss summaries & save plots of image samples to disc

#
# GENERATOR
#

# Name of generator
NAME="Simple"
# Size of random seed used by the generator's input tensor:
SEED_DIM = (32,)
# Size of the output. The generator, discriminator and dataset will be required to use this size.
IMAGE_DIM = (32, 32, 3)

def generator(input_tensor : layers.Input, input_size : Tuple[int, int, int]) -> layers.convolutional._Conv:
    # We only allow the discriminator model to work on CIFAR-sized data.
    assert input_size == (32, 32, 3)
    (img_height, img_width, img_channels) = input_size

    def add_common_layers(y):
        y = layers.Activation('relu')(y)
        return y

    #
    # input dimensions to the first conv layer in the generator
    #

    height_dim = 8
    width_dim = 8
    assert img_height % height_dim == 0 and img_width % width_dim == 0, \
        'Generator network must be able to transform `x` into a tensor of shape (img_height, img_width, img_channels).'

    # 7 * 7 * 16 == 784 input neurons
    x = layers.Dense(height_dim * width_dim * 16)(input_tensor)
    x = add_common_layers(x)

    x = layers.Reshape((height_dim, width_dim, -1))(x)

    # generator will transform `x` into a tensor w/ the desired shape by up-sampling the spatial dimension of `x`
    # through a series of strided de-convolutions (each de-conv layer up-samples spatial dim of `x` by a factor of 2).
    while height_dim != img_height:
        # spatial dim: (14 => 28 => 56 => 112 == img_height == img_width)
        height_dim *= 2
        width_dim *= 2

        # nb_feature_maps: (512 => 256 => 128 => 64)
        try:
            nb_feature_maps //= 2
        except NameError:
            nb_feature_maps = 512

        x = layers.convolutional.Conv2DTranspose(nb_feature_maps, *kernel_size,
                                                 output_shape=(None, height_dim, width_dim, nb_feature_maps),
                                                 **conv_layer_keyword_args)(x)
        x = add_common_layers(x)

    # number of feature maps => number of image channels
    return layers.convolutional.Conv2DTranspose(img_channels, 1, 1, activation='tanh',
                                                border_mode='same',
                                                output_shape=(None, img_height, img_width, img_channels))(x)


#
# DISCRIMINATOR
#

# Name of discriminator
# Each generator/discriminator needs a name if they are in different files
# NAME="Simple"

def discriminator(x, input_size):
    # We only allow the discriminator model to work on CIFAR-sized data.
    assert input_size == (32, 32, 3)
    (img_height, img_width, img_channels) = input_size

    def add_common_layers(y):
        y = layers.advanced_activations.LeakyReLU()(y)
        return y

    height_dim = 4

    # down sample with strided convolutions until we reach the desired spatial dimension (4 * 4 * nb_feature_maps)
    while x.get_shape()[1] != height_dim:
        # nb_feature_maps: (64 => 128 => 256 => 512)
        try:
            nb_feature_maps *= 2
        except NameError:
            nb_feature_maps = 64

        x = layers.convolutional.Conv2D(nb_feature_maps, *kernel_size, **conv_layer_keyword_args)(x)
        x = add_common_layers(x)

    x = layers.Flatten()(x)

    x = layers.Dense(16)(x)
    x = add_common_layers(x)

    return layers.Dense(1, activation='sigmoid')(x)