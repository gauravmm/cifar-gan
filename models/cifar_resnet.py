"""
A simple model, containing both generator and discriminator.
"""
import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

def leakyReLu(x, alpha=0.2, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)

def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))

#
# GENERATOR
#

init_kernel = None
init_bias = None

# Loosely based on https://github.com/dalgu90/resnet-18-tensorflow/blob/master/resnet.py
def res_block_head(x, out_channel, strides, is_training, name="res_block_head"):
    in_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        # Shortcut connection
        if in_channel == out_channel:
            shortcut = tf.identity(x) if strides == 1 else tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
        else:
            shortcut = tf.layers.conv2d(x, out_channel, (1, 1), strides, name='shortcut')
    return _block_inner(x, shortcut, out_channel, strides, is_training)

def res_block(x, is_training, name="res_block"):
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        return _block_inner(x, x, num_channel, 1, is_training)

def _block_inner(x, shortcut, out_channel, strides, is_training):
    # Residual
    x = tf.layers.conv2d(x, out_channel, (3, 3), strides=(strides, strides), padding="SAME", name="conv_1",
                            kernel_initializer=init_kernel,
                            bias_initializer=init_bias)
    x = tf.layers.batch_normalization(x, training=is_training, name='bn_1')
    x = leakyReLu(x)
    x = tf.layers.conv2d(x, out_channel, (3, 3), strides=1, padding="SAME", name="conv_2",
                    kernel_initializer=init_kernel,
                    bias_initializer=init_bias)
    x = tf.layers.batch_normalization(x, training=is_training, name='bn_2')
    return leakyReLu(x + shortcut)


#
# DISCRIMINATOR
#
def discriminator(inp, is_training, num_classes, **kwargs):
    x = inp

    with tf.variable_scope('conv1'):
        x = tf.layers.conv2d(x, 64, (7, 7), strides=(2, 2), padding="SAME", name="c2d2", 
                                   kernel_initializer=init_kernel,
                                   bias_initializer=init_bias)

        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyReLu(x)
        x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
    
    # conv2_x
    with tf.variable_scope('conv2'):
        x = res_block(x, is_training=is_training, name='conv2_1')
        x = res_block(x, is_training=is_training, name='conv2_2')

    # conv3_x
    with tf.variable_scope('conv3'):
        x = res_block_head(x, 128, 2, is_training=is_training, name='conv3_1')
        x = res_block(x, is_training=is_training, name='conv3_2')
    
    # conv4_x
    with tf.variable_scope('conv4'):
        x = res_block_head(x, 256, 2, is_training=is_training, name='conv4_1')
        x = res_block(x, is_training=is_training, name='conv4_2')

    # conv5_x
    with tf.variable_scope('conv5'):
        x = res_block_head(x, 512, 2, is_training=is_training, name='conv5_1')
        x = res_block(x, is_training=is_training, name='conv5_2')
    
    x = tf.contrib.layers.flatten(x)

    # The name parameters here are crucial!
    # The order of definition and inclusion in output is crucial as well! You must define y1 before y2, and also include
    # them in output in the order.
    with tf.variable_scope('discriminator'):
        y1 = tf.layers.dense(x, 16, name='fc6',
                             activation=leakyReLu,
                             kernel_initializer=init_kernel,
                             bias_initializer=init_bias)

        y1 = tf.layers.dense(y1, 1, name="fc7")
        y1 = tf.squeeze(y1, 1, name='output_node_dis')

    # Weights in scope `model_discriminator/classifier/*` are exempt from weight clipping if trained on WGANs.
    with tf.variable_scope('classifier'):
        y2 = tf.layers.dropout(x,rate=0.5,training=is_training)
        y2 = tf.layers.dense(y2, num_classes, name='output_node_cls')

    # Return (discriminator, classifier)
    return (y1, y2)


#
# GENERATOR
#
def generator(inp, is_training, inp_label, output_size, **kwargs):
    # We only allow the discriminator model to work on CIFAR-sized data.
    assert output_size == (32, 32, 3)

    # [batch_size, init_z + init_label]
    x = tf.concat([tf.contrib.layers.flatten(inp), tf.contrib.layers.flatten(inp_label)], 1)

    # [batch_size, 1, 1, init_*]
    x = tf.expand_dims(tf.expand_dims(x, 1), 1)

    # Transposed convolution outputs [batch, 4, 4, 1024]
    x = tf.layers.conv2d_transpose(x, 1024, 4, padding='valid',
            kernel_initializer=init_kernel,
            name='tconv1')
    
    x = tf.layers.batch_normalization(x, training=is_training, name='tconv1/batch_normalization')
    x = leakyReLu(x, name='tconv1/relu')
        
    # Transposed convolution outputs [batch, 8, 8, 256]
    x = tf.layers.conv2d_transpose(x, 256, 4, 2, padding='same',
            kernel_initializer=init_kernel,
            name='tconv2')
    
    x = tf.layers.batch_normalization(x, training=is_training, name='tconv2/batch_normalization')
    x = leakyReLu(x, name='tconv2/relu')
        
    # Transposed convolution outputs [batch, 16, 16, 64]
    x = tf.layers.conv2d_transpose(x, 64, 4, 2, padding='same',
            kernel_initializer=init_kernel,
            name='tconv3')
    
    x = tf.layers.batch_normalization(x, training=is_training, name='tconv3/batch_normalization')
    x = leakyReLu(x, name='tconv3/relu')
        
    # Transposed convolution outputs [batch, 32, 32, 3]
    x = tf.layers.conv2d_transpose(x, 3, 4, 2, padding='same',
            kernel_initializer=init_kernel,
            name='tconv4')
    x = tf.tanh(x, name='tconv4/tanh')

    return x