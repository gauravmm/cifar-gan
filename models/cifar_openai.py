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

init_kernel = tf.random_normal_initializer(mean = 0.0001,stddev=0.05)
init_bias = None


#the inner minibatch operation
def sigma(M):
    return tf.reduce_sum(tf.exp(-tf.reduce_mean(tf.abs(M[:,:,:]-M[:,:,:]), axis=2)), axis=1, keep_dims=True)

#matmul only supports 2D-matrix multiplication, I adapt it to 3D
def tensor_vector_multiplication(T,x):
    slices = []
    for i in range(T.get_shape().as_list()[2]):
        slices.append(tf.matmul(x,T[:,:,i]))
    return tf.stack(slices,axis=1)

#minibatch discrimination, based on OpenAI
def minibatch_disrimination(x, dim1, dim2, dim3, training=True, name='minibatch_discrimination'):
    with tf.variable_scope(name):
        T = tf.get_variable('T', [dim1,dim2,dim3], tf.float32, tf.random_normal_initializer(mean = 0.001,stddev=0.02))
        matrices = tensor_vector_multiplication(T,x)
        sigma_x = sigma(matrices)
        batch_x = tf.concat([x,sigma_x],1)
        return batch_x


#
# DISCRIMINATOR
#
def discriminator(inp, is_training, num_classes, **kwargs):
    x = inp

    with tf.variable_scope('conv1'):
        x = tf.layers.conv2d(x, 96, (3, 3), strides=(1, 1), padding="SAME", name="conv1",kernel_initializer=init_kernel,bias_initializer=init_bias)
        x = tf.leakyReLu(x)
        x = tf.layers.batch_normalization(x, training=is_training)
    with tf.variable_scope('conv2'):
        x = tf.layers.conv2d(x, 96, (3, 3), strides=(1, 1), padding="SAME", name="conv2",kernel_initializer=init_kernel,bias_initializer=init_bias)
        x = tf.leakyReLu(x)
        x = tf.layers.batch_normalization(x, training=is_training)
    with tf.variable_scope('conv3'):
        x = tf.layers.conv2d(x, 96, (3, 3), strides=(2, 2), padding="SAME", name="conv3",kernel_initializer=init_kernel,bias_initializer=init_bias)
        x = tf.leakyReLu(x)
        x = tf.layers.batch_normalization(x, training=is_training)

    x = tf.layers.dropout(x, 0.5)

    with tf.variable_scope('conv4'):
        x = tf.layers.conv2d(x, 192, (3, 3), strides=(1, 1), padding="SAME", name="conv4",kernel_initializer=init_kernel,bias_initializer=init_bias)
        x = tf.leakyReLu(x)
        x = tf.layers.batch_normalization(x, training=is_training)
    with tf.variable_scope('conv5'):
        x = tf.layers.conv2d(x, 192, (3, 3), strides=(1, 1), padding="SAME", name="conv5",kernel_initializer=init_kernel,bias_initializer=init_bias)
        x = tf.leakyReLu(x)
        x = tf.layers.batch_normalization(x, training=is_training)
    with tf.variable_scope('conv6'):
        x = tf.layers.conv2d(x, 192, (3, 3), strides=(2, 2), padding="SAME", name="conv6",kernel_initializer=init_kernel,bias_initializer=init_bias)
        x = tf.leakyReLu(x)
        x = tf.layers.batch_normalization(x, training=is_training)

    x = tf.layers.dropout(x, 0.5)

    with tf.variable_scope('conv7'):
        x = tf.layers.conv2d(x, 192, (3, 3), strides=(1, 1), padding="SAME", name="conv6",kernel_initializer=init_kernel,bias_initializer=init_bias)
        x = tf.leakyReLu(x)
        x = tf.layers.batch_normalization(x, training=is_training)

    #network-in-network layers
    with tf.variable_scope('nin1'):
        x = tf.layers.conv2d(x, 9*192, (3, 3), strides=(1, 1), padding="SAME", name="nin1-1",kernel_initializer=init_kernel,bias_initializer=init_bias)
        x = tf.leakyReLu(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.layers.conv2d(x, 192, (1, 1), strides=(1, 1), padding="SAME", name="nin1-2",kernel_initializer=init_kernel,bias_initializer=init_bias)
        x = tf.leakyReLu(x)
        x = tf.layers.batch_normalization(x, training=is_training)
    with tf.variable_scope('nin2'):
        x = tf.layers.conv2d(x, 9*192, (3, 3), strides=(1, 1), padding="SAME", name="nin2-1",kernel_initializer=init_kernel,bias_initializer=init_bias)
        x = tf.leakyReLu(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.layers.conv2d(x, 192, (1, 1), strides=(1, 1), padding="SAME", name="nin2-2",kernel_initializer=init_kernel,bias_initializer=init_bias)
        x = tf.leakyReLu(x)
        x = tf.layers.batch_normalization(x, training=is_training)

    x = tf.layers.average_pooling2d(x,8,4)
    x = minibatch_disrimination(x, 192, 32, 32, training = is_training)

    # The name parameters here are crucial!
    # The order of definition and inclusion in output is crucial as well! You must define y1 before y2, and also include
    # them in output in the order.
    with tf.variable_scope('discriminator'):
        y1 = tf.layers.dense(x, 32, name='fc6',activation=leakyReLu, kernel_initializer=init_kernel,bias_initializer=init_bias)
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

    x = tf.layers.dense(x, 4*4*512, name='fc1',activation=relu, kernel_initializer=init_kernel,bias_initializer=init_bias)

    x = tf.reshape(x, [x.get_shape().as_list()[0],512,4,4])

    # Transposed convolution outputs [batch, 8, 8, 256]
    x = tf.layers.conv2d_transpose(x, 256, 5, padding='valid',kernel_initializer=init_kernel,name='tconv1')    
    x = nn.relu(x, name='tconv1/relu')
    x = tf.layers.batch_normalization(x, training=is_training, name='tconv1/batch_normalization')

    # Transposed convolution outputs [batch, 16, 16, 128]
    x = tf.layers.conv2d_transpose(x, 128, 5, strides = (2,2), padding='valid',kernel_initializer=init_kernel,name='tconv2') 
    x = nn.relu(x, name='tconv2/relu')
    x = tf.layers.batch_normalization(x, training=is_training, name='tconv2/batch_normalization')

    # Transposed convolution outputs [batch, 32, 32, 3]
    x = tf.layers.conv2d_transpose(x, 3, 5, strides = (4,4), padding='valid',kernel_initializer=init_kernel,name='tconv3') 
    x = (tf.tanh(x, name='tconv3/tanh') + 1.0)/2

    return x