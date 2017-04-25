"""
A simple model, containing both generator and discriminator.
"""
import tensorflow as tf

from typing import Tuple

#
# GENERATOR
#

# Name of generator
NAME="Simple"
#leaky relu coefficient
ALPHA = 0.3
def leakyReLu(x, alpha=ALPHA):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))

def generator(inp, inp_label, output_size) -> Tuple[layers.convolutional._Conv, layers.convolutional._Conv]:
    # We only allow the discriminator model to work on CIFAR-sized data.
    assert output_size == (32, 32, 3)

    init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    init_gamma = tf.random_normal_initializer(mean=1.0, stddev=0.02)

    # [batch_size, init_z + init_label]
    x = tf.concat([tf.contrib.layers.flatten(inp), tf.contrib.layers.flatten(inp_label)], 1)

    # [batch_size, 1, 1, init_*]
    x = tf.expand_dims(tf.expand_dims(x, 1), 1)

    x = tf.layers.dense(x, 256, name='fc1',
                        activation=leakyReLu,
                        kernel_initializer=init_kernel,
                        bias_initializer=init_gamma)

    x = tf.reshape(x, [-1, 4, 4, -1])
    
    x = tf.layers.conv2d_transpose(x, 256, (3, 3), strides=(2, 2), name="c2t2"
                                   activation=leakyReLu,
                                   kernel_initializer=init_kernel,
                                   bias_initializer=init_gamma)
    
    x = tf.layers.conv2d_transpose(x, 128, (3, 3), strides=(2, 2), name="c2t3"
                                   activation=leakyReLu,
                                   kernel_initializer=init_kernel,
                                   bias_initializer=init_gamma)

    x = tf.layers.conv2d_transpose(x, 64, (3, 3), strides=(2, 2), name="c2t4"
                                   activation=leakyReLu,
                                   kernel_initializer=init_kernel,
                                   bias_initializer=init_gamma)

    x = tf.layers.conv2d_transpose(x, 3, (1, 1), name="c2t5"
                                   activation=tf.tanh,
                                   kernel_initializer=init_kernel,
                                   bias_initializer=init_gamma)

    return x


#
# DISCRIMINATOR
#

# Name of discriminator
# Each generator/discriminator needs a name if they are in different files
# NAME="Simple"

def discriminator(inp, num_classes):
    init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    init_gamma = tf.random_normal_initializer(mean=1.0, stddev=0.02)

    x = inp

    x = tf.layers.dense(x, 128, name='fc1',
                        kernel_initializer=init_kernel,
                        bias_initializer=init_gamma)

    x = tf.layers.conv2d(x, 128, (3, 3), strides=(2, 2), name="c2d2"
                                   activation=leakyReLu,
                                   kernel_initializer=init_kernel,
                                   bias_initializer=init_gamma)

    x = tf.layers.conv2d(x, 256, (3, 3), strides=(2, 2), name="c2d3"
                                   activation=leakyReLu,
                                   kernel_initializer=init_kernel,
                                   bias_initializer=init_gamma)

    x = tf.layers.conv2d(x, 512, (3, 3), strides=(2, 2), name="c2d4"
                                   activation=leakyReLu,
                                   kernel_initializer=init_kernel,
                                   bias_initializer=init_gamma)

    x = tf.contrib.layers.flatten(x)


    # The name parameters here are crucial!
    # The order of definition and inclusion in output is crucial as well! You must define y1 before y2, and also include
    # them in output in the order.
    with tf.name_scope('discriminator'):
        y1 = tf.layers.dense(x, 16, name='fc5'
                             activation=leakyReLu,
                             kernel_initializer=init_kernel,
                             bias_initializer=init_gamma)
        
        y1 = Dense(1, activation=tf.sigmoid, name='discriminator')(y1)
        y1 = tf.squeeze(y1)

    with tf.name_scope('classifier'):
        y2 = Dense(num_classes, activation=tf.sigmoid, name='classifier')(x)
        y2 = tf.squeeze(y2)

    # Return (discriminator, classifier)
    return (y1, y2)
