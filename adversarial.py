#!/bin/python3

"""
Main CIFAR-GAN file.
Handles the loading of data from ./data, models from ./models, training, and testing.

Modified from TensorFlow-Slim examples and https://github.com/wayaai/GAN-Sandbox/
"""

import datetime
import itertools
import logging
import math
import os
import sys
import time

import numpy as np
import png
import tensorflow as tf

import config
import support

#
# Init
#

np.random.seed(54183)

# Logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logging.basicConfig(filename=config.PATH['log'], level=logging.DEBUG, format='[%(asctime)s, %(levelname)s @%(name)s] %(message)s')
# Logger
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s @%(name)s] %(message)s', datefmt='%H:%M:%S'))
logging.getLogger().addHandler(console)
logger = logging.getLogger()

def main(args):
    """
    Main class, does:
      (1) Model building
      (2) Loading
      (3) Training
    
    args contains the commandline arguments, and the classes specified by commandline argument.
    """

    logger.info("Loaded dataset        : {}".format(args.data.__file__))
    logger.info("Loaded generator      : {}".format(args.generator.__file__))
    logger.info("Loaded discriminator  : {}".format(args.discriminator.__file__))
    logger.info("Loaded hyperparameters: {}".format(args.hyperparam.__file__))
    logger.info("Loaded preprocessors  : {}".format(", ".join(a.__file__ for a in args.preprocessor)))
    
    if args.split == "train":
        train(args)
    elif args.split == "test":
        sv = tf.train.Supervisor(logdir=config.get_filename(args), global_step=global_step, saver=None)
        with sv.managed_session() as sess:
            test(args, (gen_input, gen_label_input, gen_output, dis_input, dis_output_real_dis, dis_output_real_cls,
                        dis_output_fake_dis, dis_output_fake_cls))
    else:
        assert not "This state should not be reachable; the argparser should catch this case."

_shape_str = lambda a: "(" + ", ".join("?" if b is None else str(b) for b in a) + ")"
def train(args):
    logger = logging.getLogger("train")

    #
    # Build Model
    #
    
    global_step = tf.Variable(0, name='global_step', trainable=False)

    gen_input  = tf.placeholder(tf.float32, shape=[None] + list(args.hyperparam.SEED_DIM), name="input_gen_seed")
    gen_label_input  = tf.placeholder(tf.float32, shape=(None, args.hyperparam.NUM_CLASSES), name="input_gen_class")
    with tf.variable_scope('model_generator'):
        gen_output = args.generator.generator(gen_input, gen_label_input, args.hyperparam.IMAGE_DIM)
        if not str(gen_output.get_shape()) == _shape_str([None] + list(args.hyperparam.IMAGE_DIM)):
            logger.error("Generator output size is incorrect! Expected: {}, actual: {}".format(
                    str(gen_output.get_shape()), _shape_str([None] + list(args.hyperparam.IMAGE_DIM))))

    dis_input  = tf.placeholder(tf.float32, shape=[None] + list(args.hyperparam.IMAGE_DIM), name="input_dis_image")
    dis_label  = tf.placeholder(tf.float32, shape=[None, 1], name="input_dis_label")
    dis_class  = tf.placeholder(tf.float32, shape=(None, args.hyperparam.NUM_CLASSES), name="input_dis_class")
    with tf.variable_scope('model_discriminator') as disc_scope:
        dis_output_real_dis, dis_output_real_cls = args.discriminator.discriminator(dis_input, args.hyperparam.NUM_CLASSES)
        disc_scope.reuse_variables()
        dis_output_fake_dis, dis_output_fake_cls = args.discriminator.discriminator(gen_output, args.hyperparam.NUM_CLASSES)

    # Discriminator losses
    dis_loss_real = tf.reduce_mean(tf.nn.l2_loss(dis_output_real_dis - dis_label))
    dis_loss_fake = tf.reduce_mean(tf.nn.l2_loss(dis_output_fake_dis - dis_label))

    # Classifier loss
    cls_loss_dis  = tf.reduce_mean(tf.nn.l2_loss(dis_output_fake_dis - dis_label))
    cls_loss_cls  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=dis_label, logits=dis_output_real_cls))
    cls_loss = args.hyperparam.loss_weights_classifier["discriminator"] * cls_loss_dis \
             + args.hyperparam.loss_weights_classifier["classifier"]    * cls_loss_cls

    # Generator loss
    gen_loss_dis  = tf.reduce_mean(tf.nn.l2_loss(dis_output_fake_dis - dis_label))
    gen_loss_cls  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gen_label_input, logits=dis_output_fake_cls))
    gen_loss = args.hyperparam.loss_weights_generator["discriminator"] * gen_loss_dis \
             + args.hyperparam.loss_weights_generator["classifier"]    * gen_loss_cls

    logger.info("Model constructed.")
    data = support.TrainData(args)

    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    discriminator_variables = [v for v in variables if 'model_discriminator/' in v.name]
    generator_variables     = [v for v in variables if 'model_generator/' in v.name]

    # Train ops
    train_dis_real = args.hyperparam.optimizer_dis. \
                          minimize(dis_loss_real, var_list=discriminator_variables)
    train_dis_fake = args.hyperparam.optimizer_dis. \
                          minimize(dis_loss_fake, var_list=discriminator_variables)
    train_cls = args.hyperparam.optimizer_cls. \
                          minimize(gen_loss, var_list=discriminator_variables)
    train_gen = args.hyperparam.optimizer_gen. \
                          minimize(gen_loss, var_list=generator_variables)

    # Prepare summaries, in order of train loss above:
    assert support.Y_REAL == 0 and support.Y_FAKE == 1
    with tf.name_scope('summary'):
        with tf.name_scope('discriminator'):
            tf.summary.scalar('fake/loss', dis_loss_fake)
            train_dis_fake_true_neg = tf.reduce_sum(tf.cast(tf.greater_equal(dis_output_fake_dis, 0.5), tf.int32))
            tf.summary.scalar('fake/true_neg', train_dis_fake_true_neg)

            tf.summary.scalar('real/loss', dis_loss_real)
            train_dis_real_true_pos = tf.reduce_sum(tf.cast(tf.less(dis_output_real_dis, 0.5), tf.int32))
            tf.summary.scalar('real/true_pos', train_dis_real_true_pos)
            
        with tf.name_scope('classifier'):
            tf.summary.scalar('true_pos', tf.reduce_sum(tf.cast(tf.less(dis_output_fake_dis, 0.5), tf.int32)))
    
        with tf.name_scope('generator'):
            tf.summary.image('output', data.unapply(gen_output), max_outputs=8)
            tf.summary.scalar('loss', gen_loss)
            tf.summary.scalar('fooling_rate', tf.reduce_sum(tf.cast(tf.less(dis_output_fake_dis, 0.5), tf.int32)))

        with tf.name_scope('iterations'):
            # Create placeholder variables
            log_step_dis = tf.Variable(0, trainable=False)
            log_step_cls = tf.Variable(0, trainable=False)
            log_step_gen = tf.Variable(0, trainable=False)

            tf.summary.scalar('discriminator', log_step_dis)
            tf.summary.scalar('classifier', log_step_cls)
            tf.summary.scalar('generator', log_step_gen)

    sv = tf.train.Supervisor(logdir=config.get_filename(".", args), global_step=global_step, save_summaries_secs=60, save_model_secs=args.log_interval)
    with sv.managed_session() as sess:
        logger.info("Starting training. Saving model every {}s.".format(args.log_interval))

        # Format the score printing
        while sv.should_stop():
            batch = sess.run(global_step)
            logger.debug('Step {} of {}.'.format(batch, args.batches))

            # This training proceeds in two phases: (1) discriminator, then (2) generator.
            # First, we train the discriminator for `step_dis` number of steps. Because the .trainable flag was True when 
            # `dis_model` was compiled, the weights of the discriminator will be updated. The discriminator is trained to
            # distinguish between "fake"" (generated) and real images by running it on one step of each.
            
            step_dis = 0
            while True:
                # Generate fake images, and train the model to predict them as fake. We keep track of the loss in predicting
                # fake images separately from real images.
                loss_fake, fake_true_neg = sess.run([train_dis_fake, train_dis_fake_true_neg], feed_dict={
                    gen_input: next(data.rand_vec),
                    gen_label_input: next(data.rand_label_vec),
                    dis_label: next(data.label_dis_fake)
                })
                
                # Use real images (but not labels), and train the model to predict them as real.
                loss_real, real_true_pos = sess.run([train_dis_real, train_dis_real_true_pos], feed_dict={
                    dis_input: next(data.unlabelled),
                    dis_label: next(data.label_dis_real)
                })
                
                step_dis += 1
                if args.hyperparam.discriminator_halt(batch, step_dis, 
                    {"fake_loss": loss_fake, "fake_true_neg": fake_true_neg,
                     "real_loss": loss_real, "real_true_pos": real_true_pos}):
                    break
            # Log in tensorboard
            sess.run(tf.assign(log_step_dis, step_dis))

            # Train classifier
            step_cls = 0
            while True:
                data_x, data_y = next(data.labelled)
                loss_label = dis_model_labelled.train_on_batch(data_x, [ next(data.label_dis_real), data_y ])
                step_cls += 1
                if args.hyperparam.classifier_halt(batch, step_cls):
                    break


            # Second, we train the generator for `step_gen` number of steps. Because the .trainable flag (for `dis_model`) 
            # was False when `com_model` was compiled, the discriminator weights are not updated. The generator weights are
            # the only weights updated in this step.
            # In this step, we train "generator" so that "discriminatsor(generator(random)) == real". Specifically, we compose
            # `dis_model` onto `gen_model`, and train the combined model so that given a random vector, it classifies images
            # as real.
            step_com = 0
            while True:
                fake_class = next(data.rand_label_vec)
                loss = com_model.train_on_batch(
                    [ next(data.rand_vec), fake_class ],
                    [ next(data.label_gen_real), fake_class ])
                
                step_com += 1
                if args.hyperparam.generator_halt(batch, step_com, loss):
                    break

            #
            # That is the entire training algorithm.
            #
            sess.run(tf.assign_add(global_step, 1))


#
# Training
#

def test(args, metrics_names, models):
    logger = logging.getLogger("test")

    dis_model_unlabelled, dis_model_labelled, gen_model, com_model = models
    VERIFY_METRIC = True
    if VERIFY_METRIC:
        logger.warn("VERIFY_METRIC is True; computation will take longer.")
    
    metric_wrap = lambda x: {k:v for k, v in zip(metrics_names, x)}
    data = support.TestData(args, "test")

    logger.info("Starting tests.")
    metrics = np.zeros(shape=len(metrics_names))
    i = 0
    q = 0.0
    k = None
    for batch, d in enumerate(data.labelled):
        data_x, data_y = d
        m = dis_model_labelled.test_on_batch(data_x, [next(data.label_dis_real), data_y])
        metrics += m
        i += 1
        if VERIFY_METRIC:
            v = dis_model_labelled.predict(data_x)[1]
            q += np.sum(np.argmax(v, axis=1) == np.argmax(data_y, axis=1))/v.shape[0]
            if k is None:
                k = np.zeros((v.shape[1], v.shape[1]))
            for (x, y) in zip(np.argmax(data_y, axis=1), np.argmax(v, axis=1)):
                k[x, y] += 1.0/v.shape[0]

    metrics /= i
    m = metric_wrap(metrics)
    logger.debug(m)
    logger.info("Classifier Accuracy: {:.1f}%".format(m['classifier_acc']*100))
    if VERIFY_METRIC:
        logger.info("Classifier Accuracy (compare): {:.1f}%".format(q/i*100))
        k = k/i
        k = k/np.sum(k, axis=0)*100.0
        logger.info("Confusion Matrix [Actual, Reported] (%):\n" + np.array_str(k, max_line_width=120, precision=1, suppress_small=True))
    logger.info("Discriminator Recall: {:.1f}%".format(m['discriminator_label_real']*100))


if __name__ == '__main__':
    logger.info("Started")
    try:
        main(support.argparser().parse_args())
    except:
        raise
    finally:
        logger.info("Halting")