"""
A simple model, containing both generator and discriminator.
"""
import logging

import tensorflow as tf

from models import openai_tf_weightnorm as otw
from hyperparam import simple


logger = logging.getLogger(__name__)

def leakyReLu(x, alpha=0.2, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)

def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))

#the inner minibatch operation
def sigma(M, N, dim2, dim3):
    M.set_shape([simple.BATCH_SIZE,dim2,dim3])
    N.set_shape([simple.BATCH_SIZE,dim2,dim3])
    l = []
    for i in range(simple.BATCH_SIZE):
        l.append(tf.exp(-tf.reduce_mean(tf.abs(M[:,:,:]-N[i,:,:]), axis=2)))
    stacked = tf.stack(l, axis=0)
    stacked_summed = tf.reduce_sum(stacked, axis=1)
    stacked_summed.set_shape([None,dim2])
    return stacked_summed

#matmul only supports 2D-matrix multiplication, I adapt it to 3D
def tensor_vector_multiplication(T,x):
    slices = []
    for i in range(T.get_shape().as_list()[2]):
        slices.append(tf.matmul(x,T[:,:,i]))
    return tf.stack(slices,axis=1)

#minibatch discrimination, based on OpenAI
def minibatch_disrimination(x, dim1, dim2, dim3, name='minibatch_discrimination'):
    with tf.variable_scope(name):
        T = tf.Variable(name='T', dtype=tf.float32, initial_value=tf.random_normal([dim1,dim2,dim3], 0.001, 0.05))
        matrices = tensor_vector_multiplication(T,x)
        sigma_x = sigma(matrices, matrices, dim2, dim3)
        batch_x = tf.concat([x,sigma_x],1)
        return batch_x

init_kernel = tf.random_normal_initializer(mean = 0.0001,stddev=0.05)
init_bias = None


#
# DISCRIMINATOR
#
def discriminator(inp, is_training, num_classes, **kwargs):
    x = inp
    print(x.get_shape())

    with tf.variable_scope('conv1'):
        x = otw.conv2d(x, 96, nonlinearity=leakyReLu, name='conv1', init=True)
    with tf.variable_scope('conv2'):
        x = otw.conv2d(x, 96, nonlinearity=leakyReLu, name = "conv2", init=True)
    with tf.variable_scope('conv3'):
        x = otw.conv2d(x, 96, nonlinearity=leakyReLu, stride=[2,2], name = "conv3", init=True)
    print(x.get_shape())

    x = tf.layers.dropout(x, 0.5)

    with tf.variable_scope('conv4'):
        x = otw.conv2d(x, 192, nonlinearity=leakyReLu, name = "conv4", init=True)
    with tf.variable_scope('conv5'):
        x = otw.conv2d(x, 192, nonlinearity=leakyReLu, name = "conv5", init=True)
    with tf.variable_scope('conv6'):
        x = otw.conv2d(x, 192, nonlinearity=leakyReLu, stride=[2,2], name = "conv6", init=True)
    print(x.get_shape())

    x = tf.layers.dropout(x, 0.5)

    with tf.variable_scope('conv7'):
        x = otw.conv2d(x, 192, nonlinearity=leakyReLu, name = "conv7", init=True)
    print(x.get_shape())

    #network-in-network layers
    with tf.variable_scope('nin1'):
        x = otw.nin(x, 192, name="nin1", init=True)
    print(x.get_shape())
    with tf.variable_scope('nin2'):
        x = otw.nin(x, 192, name="nin2", init=True)
    print(x.get_shape())

    x = tf.layers.average_pooling2d(x,5,4)
    print(x.get_shape())
    x = tf.squeeze(x, [1,2])
    print(x.get_shape())
    x = minibatch_disrimination(x, 192, 10, 10)
    print(x.get_shape())

    # The name parameters here are crucial!
    # The order of definition and inclusion in output is crucial as well! You must define y1 before y2, and also include
    # them in output in the order.
    with tf.variable_scope('discriminator'):
        y1 = otw.dense(x, 10, name='fc1', init=True)
        print(y1.get_shape())
        y1 = otw.dense(y1, 1, name='fc2', init=True)
        print(y1.get_shape())
        y1 = tf.squeeze(y1, 1, name='output_node_dis')
        print(y1.get_shape())

    # Weights in scope `model_discriminator/classifier/*` are exempt from weight clipping if trained on WGANs.
    with tf.variable_scope('classifier'):
        y2 = tf.layers.dropout(x,rate=0.5, training=is_training)
        print(y2.get_shape())
        y2 = otw.dense(x, num_classes, name='output_node_cls', init=True)
        print(y2.get_shape())

    # Return (discriminator, classifier)
    return (y1, y2)


#
# GENERATOR
#
def generator(inp, is_training, inp_label, output_size, **kwargs):
    # We only allow the discriminator model to work on CIFAR-sized data.
    assert output_size == (32, 32, 3)
    
    print(inp.get_shape())
    # [batch_size, init_z + init_label]
    x = tf.concat([tf.contrib.layers.flatten(inp), tf.contrib.layers.flatten(inp_label)], 1)
    print(x.get_shape())

    with tf.variable_scope('dense4'):
        x = tf.layers.dense(x, 4*4*512, name='fc4', kernel_initializer=init_kernel)
        x = tf.layers.batch_normalization(x, training=is_training, name='fc4/batch_norm')
        x = tf.nn.relu(x)
    print(x.get_shape())

    x = tf.reshape(x, [-1,4,4,512])
    print(x.get_shape())

    # Transposed convolution outputs [batch, 8, 8, 256]
    with tf.variable_scope('deconv1'):
        W = tf.get_variable('W', [5,5,256,512], tf.float32, init_kernel)        
        batch_size = tf.shape(x)[0]
        output_shape = tf.stack([batch_size, tf.shape(x)[1]*2, tf.shape(x)[2]*2, 256])
        temp1 = x.get_shape().as_list()[1]
        temp2 = x.get_shape().as_list()[2]
        x = tf.nn.conv2d_transpose(x, W, output_shape, [1,2,2,1], padding='SAME', name="deconv1")
        real_output_shape = tf.stack([batch_size, temp1*2, temp2*2, 256])
        x = tf.reshape(x, real_output_shape)
        x = tf.layers.batch_normalization(x, training=is_training, name='deconv1/batch_norm')
        x = tf.nn.relu(x)  
    print(x.get_shape())

    # Transposed convolution outputs [batch, 16, 16, 128]
    with tf.variable_scope('deconv2'):
        W = tf.get_variable('W', [5,5,128,256], tf.float32, init_kernel)
        batch_size = tf.shape(x)[0]
        output_shape = tf.stack([batch_size, tf.shape(x)[1]*2, tf.shape(x)[2]*2, 128])
        temp1 = x.get_shape().as_list()[1]
        temp2 = x.get_shape().as_list()[2]
        x = tf.nn.conv2d_transpose(x, W, output_shape, [1,2,2,1], padding='SAME', name="deconv2")
        real_output_shape = tf.stack([batch_size, temp1*2, temp2*2, 128])
        x = tf.reshape(x, real_output_shape)
        x = tf.layers.batch_normalization(x, training=is_training, name='deconv2/batch_norm')
        x = tf.nn.relu(x)    
    print(x.get_shape())

    # Transposed convolution outputs [batch, 32, 32, 3]
    with tf.variable_scope('generator'):
        x = otw.deconv2d(x, 3, filter_size=[5,5], stride=[2,2], nonlinearity=tf.tanh, name = "output_node_gen", init=True)
        print(x.get_shape())
        return x