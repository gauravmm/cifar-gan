"""
A simple model, containing both generator and discriminator.
"""
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

#
# GENERATOR
#

_conv3d_transpose_num = 0
def conv3d_transpose(inp, new_channels, kernel_size, strides=(1, 1, 1), activation=None, padding="SAME", kernel_initializer=None, bias_initializer=None, name=None):
    # Get the input dimensions
    b, x, y, z, c = inp.get_shape().as_list()
    i, j, k = kernel_size
    si, sj, sk = strides

    # logger.warn((numel, x, y, z, c), (i,j,k))
    
    if name is None:
        name = "c3d_t_{}".format(_conv3d_transpose_num)
        _conv3d_transpose_num += 1

    if kernel_initializer is None:
        kernel_initializer = tf.zeros_initializer()
    if bias_initializer is None:
        bias_initializer = tf.zeros_initializer()

    if i % 2 == 0 or j % 2 == 0 or k % 2 == 0:
        logger.error("Your kernel size is not odd. conv3d_t is not tested to be compatible with this.")
        raise AssertionError()
    
    if si > i or sj > j or sj > j:
        logger.warn("The stride length is greater than the kernel size. Your output will have a blank grid pattern.")

    with tf.variable_scope(name + "_vars"):
        if padding == "SAME":
            # depth, height, width, output_channels, in_channels
            kernel = tf.Variable(initial_value=kernel_initializer((i, j, k, new_channels, c), inp.dtype))
            
            # There are many possible output shapes that are consistent with the stride, kernel, and input size, because
            # alignment information is lost in the rounding down. We just pick the simplest possible output shape.
            
            # We also compute this shape for each batch, so we can (in the future) support non-uniform batch sizes.
            output_shape = tf.stack((tf.shape(inp)[0], x*si, y*sj, z*sk, new_channels))
            # output_shape = (-1, x*si+i-1, y*sj+j-1, z*sk+k-1, new_channels)

        elif padding == "VALID":
            raise AssertionError("Only SAME padding implemented for conv3d_transpose.")
        else:
            raise AssertionError("Padding must be SAME or VALID for conv3d_transpose.")

    rv = tf.nn.conv3d_transpose(inp, kernel, output_shape, strides=(1, si, sj, sk, 1), padding=padding, name=name)
    
    if activation is not None and callable(activation):
        return activation(rv)
    
def leakyReLu(x, alpha=0.3):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))

def generator(inp, inp_label, output_size):
    # We only allow the discriminator model to work on CIFAR-sized data.
    assert output_size == (32, 32, 32, 1)

    init_kernel = None
    init_bias = None

    # [batch_size, init_z + init_label]
    x = tf.concat([tf.contrib.layers.flatten(inp), tf.contrib.layers.flatten(inp_label)], 1)

    # [batch_size, 1, 1, init_*]
    x = tf.expand_dims(tf.expand_dims(x, 1), 1)
    
    x = tf.layers.dense(x, 32 * 2*2*2, name='fc1',
                        activation=leakyReLu,
                        kernel_initializer=init_kernel,
                        bias_initializer=init_bias)

    x = tf.expand_dims(x, 4)

    x = tf.reshape(x, [-1, 2, 2, 2, 32])

    x = conv3d_transpose(x, 16, (3, 3, 3), strides=(2, 2, 2), name="c3t2",
                                   activation=leakyReLu,
                                   kernel_initializer=init_kernel,
                                   bias_initializer=init_bias)

    x = conv3d_transpose(x, 8, (3, 3, 3), strides=(2, 2, 2), name="c3t3",
                                   activation=leakyReLu,
                                   kernel_initializer=init_kernel,
                                   bias_initializer=init_bias)

    x = conv3d_transpose(x, 4, (3, 3, 3), strides=(2, 2, 2), name="c3t4",
                                   activation=leakyReLu,
                                   kernel_initializer=init_kernel,
                                   bias_initializer=init_bias)

    x = conv3d_transpose(x, 1, (3, 3, 3), strides=(2, 2, 2), name="c3t5",
                                   activation=leakyReLu,
                                   kernel_initializer=init_kernel,
                                   bias_initializer=init_bias)

    return x


#
# DISCRIMINATOR
#

def discriminator(inp, num_classes):
    init_kernel = None
    init_bias = None

    x = inp

    x = tf.layers.conv3d(x, 8, (3, 3, 3), strides=(2, 2, 2), padding="SAME", name="c2d2",
                         activation=leakyReLu,
                         kernel_initializer=init_kernel,
                         bias_initializer=init_bias)

    x = tf.layers.conv3d(x, 4, (3, 3, 3), strides=(2, 2, 2), padding="SAME", name="c2d3",
                         activation=leakyReLu,
                         kernel_initializer=init_kernel,
                         bias_initializer=init_bias)

    x = tf.layers.conv3d(x, 2, (3, 3, 3), strides=(2, 2, 2), padding="SAME", name="c2d4",
                         activation=leakyReLu,
                         kernel_initializer=init_kernel,
                         bias_initializer=init_bias)

    x = tf.contrib.layers.flatten(x)

    # The name parameters here are crucial!
    # The order of definition and inclusion in output is crucial as well! You must define y1 before y2, and also include
    # them in output in the order.
    with tf.variable_scope('discriminator'):
        y1 = tf.layers.dense(x, 16, name='fc5',
                             activation=leakyReLu,
                             kernel_initializer=init_kernel,
                             bias_initializer=init_bias)

        y1 = tf.layers.dense(y1, 1, name="fc6")
        y1 = tf.squeeze(y1, 1, name='output_node_dis')

    # Weights in scope `model_discriminator/classifier/*` are exempt from weight clipping if trained on WGANs.
    with tf.variable_scope('classifier'):
        y2 = tf.layers.dense(x, num_classes, name='output_node_cls')

    # Return (discriminator, classifier)
    return (y1, y2)
