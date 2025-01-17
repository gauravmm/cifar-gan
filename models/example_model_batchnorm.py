"""
A simple model, containing both generator and discriminator.
"""
import tensorflow as tf

#
# GENERATOR
#

def leakyReLu(x, alpha=0.3):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))

def generator(inp, is_training, inp_label, output_size, **kwargs):
    # We only allow the discriminator model to work on CIFAR-sized data.
    assert output_size == (32, 32, 3)

    init_kernel = None
    init_bias = None

    # [batch_size, init_z + init_label]
    x = tf.concat([tf.contrib.layers.flatten(inp), tf.contrib.layers.flatten(inp_label)], 1)

    # [batch_size, 1, 1, init_*]
    x = tf.expand_dims(tf.expand_dims(x, 1), 1)

    x = tf.layers.dense(x, 256, name='fc1',
                        activation=leakyReLu,
                        kernel_initializer=init_kernel,
                        bias_initializer=init_bias)

    x = tf.reshape(x, [-1, 4, 4, 16])

    x = tf.layers.conv2d_transpose(x, 256, (3, 3), strides=(2, 2), padding="SAME", name="c2t2",
                                activation=leakyReLu,
                                kernel_initializer=init_kernel,
                                bias_initializer=init_bias)

    x = tf.layers.conv2d_transpose(x, 128, (3, 3), strides=(2, 2), padding="SAME", name="c2t3",
                                activation=leakyReLu,
                                kernel_initializer=init_kernel,
                                bias_initializer=init_bias)

    x = tf.layers.conv2d_transpose(x, 64, (3, 3), strides=(2, 2), padding="SAME", name="c2t4",
                                activation=leakyReLu,
                                kernel_initializer=init_kernel,
                                bias_initializer=init_bias)

    x = tf.layers.conv2d_transpose(x, 3, (1, 1), padding="SAME", name="c2t5",
                                activation=tf.tanh,
                                kernel_initializer=init_kernel,
                                bias_initializer=init_bias)

    return x


#
# DISCRIMINATOR
#

def discriminator(inp, is_training, num_classes, **kwargs):
    init_kernel = None
    init_bias = None
    batch_normalization = kwargs['batch_norm']

    x = inp

    x = tf.layers.dense(x, 64, name='fc1',
                        kernel_initializer=init_kernel,
                        bias_initializer=init_bias)

    x = batch_normalization(x, training=is_training)

    x = tf.layers.conv2d(x, 128, (3, 3), strides=(2, 2), padding="SAME", name="c2d2", 
                                   activation=leakyReLu,
                                   kernel_initializer=init_kernel,
                                   bias_initializer=init_bias)

    x = batch_normalization(x, training=is_training)

    x = tf.layers.conv2d(x, 256, (3, 3), strides=(2, 2), padding="SAME", name="c2d3",
                                   activation=leakyReLu,
                                   kernel_initializer=init_kernel,
                                   bias_initializer=init_bias)

    x = batch_normalization(x, training=is_training, name="bn4")
    
    x = tf.layers.conv2d(x, 512, (3, 3), strides=(2, 2), padding="SAME", name="c2d5",
                                   activation=leakyReLu,
                                   kernel_initializer=init_kernel,
                                   bias_initializer=init_bias)

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
        y2 = tf.layers.dense(x, num_classes, name='output_node_cls')

    # Return (discriminator, classifier)
    return (y1, y2)
