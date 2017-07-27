"""
Same as cifar_resnet_original.py but the end of D and C is adapted to only train a good C.
This Resnet can achieve 80% accuracy when trained in a supervised way, and without data augmentation.
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
        x = tf.layers.conv2d(x, 16, (3, 3), strides=(1, 1), padding="SAME", name="c2d2", 
                                   kernel_initializer=init_kernel,
                                   bias_initializer=init_bias)

        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyReLu(x)
        #x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
    
    # conv2_x
    with tf.variable_scope('conv2'):
        x = res_block_head(x, 16, 1, is_training=is_training, name='conv2_1')
        x = res_block(x, is_training=is_training, name='conv2_2')
        x = res_block(x, is_training=is_training, name='conv2_3')

    # conv3_x
    with tf.variable_scope('conv3'):
        x = res_block_head(x, 32, 2, is_training=is_training, name='conv3_1')
        x = res_block(x, is_training=is_training, name='conv3_2')
        x = res_block(x, is_training=is_training, name='conv3_3')

    # conv4_x
    with tf.variable_scope('conv4'):
        x = res_block_head(x, 64, 2, is_training=is_training, name='conv4_1')
        x = res_block(x, is_training=is_training, name='conv4_2')
        x = res_block(x, is_training=is_training, name='conv4_3')
    
    x = tf.layers.average_pooling2d(x,8,4)

    # The name parameters here are crucial!
    # The order of definition and inclusion in output is crucial as well! You must define y1 before y2, and also include
    # them in output in the order.
    with tf.variable_scope('discriminator'):
        # with tf.variable_scope('conv6_dis'):
        #     y1 = res_block_head(x, 128, 1, is_training=is_training, name='conv6_1')
        #     y1 = res_block(y1, is_training=is_training, name='conv6_2')  
        # y1 = tf.layers.average_pooling2d(y1,8,4)
        y1 = tf.contrib.layers.flatten(x)
        y1 = tf.layers.dense(y1, 32, name='fc6',activation=leakyReLu,kernel_initializer=init_kernel,bias_initializer=init_bias)
        y1 = tf.layers.dense(y1, 1, name="fc7")
        y1 = tf.squeeze(y1, 1, name='output_node_dis')

    # Weights in scope `model_discriminator/classifier/*` are exempt from weight clipping if trained on WGANs.
    with tf.variable_scope('classifier'):
        # with tf.variable_scope('conv6_cls'):
        #     y2 = res_block_head(x, 128, 1, is_training=is_training, name='conv6_1')
        #     y2 = res_block(y2, is_training=is_training, name='conv6_2') 
        # y2 = tf.layers.average_pooling2d(y2,8,4)
        y2 = tf.contrib.layers.flatten(x)
        # y2 = tf.layers.dense(y2, 128, name='fc8',activation=leakyReLu,kernel_initializer=init_kernel,bias_initializer=init_bias)
        # y2 = tf.layers.dropout(y2,rate=0.5,training=is_training)
        y2 = tf.layers.dense(y2, num_classes, name='output_node_cls')

    # Return (discriminator, classifier)
    return (y1, y2)


#
# GENERATOR
#
#we don't care about the G here actually...
