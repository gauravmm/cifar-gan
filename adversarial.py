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

import config
import support
import tensorflow as tf

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
    
    run(args)


_shape_str = lambda a: "(" + ", ".join("?" if b is None else str(b) for b in a) + ")"
def run(args):
    logger = logging.getLogger("train")

    #
    # Build Model
    #
    
    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False, dtype=tf.int32)

    gen_input_seed  = tf.placeholder(tf.float32, shape=[None] + list(args.hyperparam.SEED_DIM), name="input_gen_seed")
    gen_input_class  = tf.placeholder(tf.float32, shape=(None, args.hyperparam.NUM_CLASSES), name="input_gen_class")
    with tf.variable_scope('model_generator'):
        gen_output = args.generator.generator(gen_input_seed, gen_input_class, args.hyperparam.IMAGE_DIM)
        
        # Sanity checking output
        if not str(gen_output.get_shape()) == _shape_str([None] + list(args.hyperparam.IMAGE_DIM)):
            logger.error("Generator output size is incorrect! Expected: {}, actual: {}".format(
                    _shape_str([None] + list(args.hyperparam.IMAGE_DIM)), str(gen_output.get_shape())))

    dis_input      = tf.placeholder(tf.float32, shape=[None] + list(args.hyperparam.IMAGE_DIM), name="input_dis_image")
    dis_label_real = tf.placeholder(tf.float32, shape=[None], name="input_dis_label")
    dis_label_fake = tf.placeholder(tf.float32, shape=[None], name="input_dis_label")
    dis_class      = tf.placeholder(tf.float32, shape=(None, args.hyperparam.NUM_CLASSES), name="input_dis_class")
    with tf.variable_scope('model_discriminator') as disc_scope:
        # Make sure that the generator and real images are the same size:
        assert str(gen_output.get_shape()) == str(dis_input.get_shape())
        dis_output_real_dis, dis_output_real_cls = args.discriminator.discriminator(dis_input, args.hyperparam.NUM_CLASSES)
        disc_scope.reuse_variables()
        dis_output_fake_dis, dis_output_fake_cls = args.discriminator.discriminator(gen_output, args.hyperparam.NUM_CLASSES)

        # Sanity checking output
        if not str(dis_output_real_dis.get_shape()) == "(?,)" or \
           not str(dis_output_real_dis.get_shape()) == "(?,)":
            logger.error("Discriminator dis (y1) output size is incorrect! Expected: {}, actual: {} and {}".format(
                "(?,)",
                str(dis_output_real_dis.get_shape()),
                str(dis_output_real_dis.get_shape())))
            return

        if not str(dis_output_real_cls.get_shape()) == _shape_str([None, args.hyperparam.NUM_CLASSES]) or \
           not str(dis_output_fake_cls.get_shape()) == _shape_str([None, args.hyperparam.NUM_CLASSES]):
            logger.error("Discriminator cls (y2) output size is incorrect! Expected: {}, actual: {} and {}".format(
                _shape_str([None]),
                str(dis_output_real_cls.get_shape()),
                str(dis_output_fake_cls.get_shape())))
            return

    with tf.name_scope('metrics'):
        # Discriminator losses
        with tf.name_scope('dis_fake'):
            with tf.name_scope('loss'):
                dis_loss_fake = tf.reduce_mean(tf.nn.l2_loss(dis_output_fake_dis - dis_label_fake))
            with tf.name_scope('true_neg'):
                train_dis_fake_true_neg = tf.reduce_mean(tf.cast(tf.greater_equal(dis_output_fake_dis, 0.5), tf.int32))
        with tf.name_scope('dis_real'):
            with tf.name_scope('loss'):
                dis_loss_real = tf.reduce_mean(tf.nn.l2_loss(dis_output_real_dis - dis_label_real))
            with tf.name_scope('true_pos'):
                train_dis_real_true_pos = tf.reduce_mean(tf.cast(tf.less(dis_output_real_dis, 0.5), tf.int32))
                
        # Classifier loss
        with tf.name_scope('cls'):
            with tf.name_scope('dis'):
                cls_loss_dis  = tf.reduce_mean(tf.nn.l2_loss(dis_output_real_dis - dis_label_real))
            with tf.name_scope('cls'):
                cls_loss_cls  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=dis_class, logits=dis_output_real_cls))
            cls_loss = args.hyperparam.loss_weights_classifier["discriminator"] * cls_loss_dis \
                    + args.hyperparam.loss_weights_classifier["classifier"]    * cls_loss_cls
            
            with tf.name_scope('acc'):
                cls_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(dis_class, axis=1), tf.argmax(dis_output_real_cls, axis=1)), tf.int32))

        # Generator loss
        with tf.name_scope('gen'):
            with tf.name_scope('dis'):
                gen_loss_dis  = tf.reduce_mean(tf.nn.l2_loss(dis_output_fake_dis - dis_label_real))
            with tf.name_scope('cls'):
                gen_loss_cls  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gen_input_class, logits=dis_output_fake_cls))
            gen_loss = args.hyperparam.loss_weights_generator["discriminator"] * gen_loss_dis \
                     + args.hyperparam.loss_weights_generator["classifier"]    * gen_loss_cls
            
            with tf.name_scope("fooling_rate"):
                gen_fooling = tf.reduce_mean(tf.cast(tf.less(dis_output_fake_dis, 0.5), tf.int32))


        logger.info("Model constructed.")

    # Preprocessor
    preproc = support.Preprocessor(args)

    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    discriminator_variables = [v for v in variables if 'model_discriminator/' in v.name]
    generator_variables     = [v for v in variables if 'model_generator/' in v.name]

    # Train ops
    with tf.name_scope('train_ops'):
        train_dis_fake = args.hyperparam.optimizer_dis. \
                            minimize(dis_loss_fake, var_list=discriminator_variables)
        train_dis_real = args.hyperparam.optimizer_dis. \
                            minimize(dis_loss_real, var_list=discriminator_variables)
        train_cls = args.hyperparam.optimizer_cls. \
                            minimize(cls_loss, var_list=discriminator_variables)
        train_gen = args.hyperparam.optimizer_gen. \
                            minimize(gen_loss, var_list=generator_variables)

    with tf.name_scope('step_count'):

        log_step_dis_val = tf.placeholder(tf.int32, shape=())
        log_step_dis = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
        log_step_dis_assign = tf.assign(log_step_dis, log_step_dis_val)
                
        log_step_cls_val = tf.placeholder(tf.int32, shape=())
        log_step_cls = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
        log_step_cls_assign = tf.assign(log_step_cls, log_step_cls_val)
        
        log_step_gen_val = tf.placeholder(tf.int32, shape=())
        log_step_gen = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
        log_step_gen_assign = tf.assign(log_step_gen, log_step_gen_val)

    # Prepare summaries, in order of train loss above:
    assert support.Y_REAL == 0 and support.Y_FAKE == 1

    with tf.name_scope('summary_discriminator'):
        with tf.name_scope('fake'):
            tf.summary.scalar('loss', dis_loss_fake)
            tf.summary.scalar('true_neg', train_dis_fake_true_neg)
            tf.summary.histogram('dis', dis_output_fake_dis)

        with tf.name_scope('real'):
            tf.summary.scalar('loss', dis_loss_real)
            tf.summary.scalar('true_pos', train_dis_real_true_pos)
            tf.summary.histogram('dis', dis_output_real_dis)
        
    with tf.name_scope('summary_classifier'):
        tf.summary.scalar('loss/cls', cls_loss_cls)
        tf.summary.scalar('loss/dis', cls_loss_dis)
        tf.summary.scalar('loss', cls_loss)
        tf.summary.scalar('accuracy', cls_accuracy)
        tf.summary.histogram('label_actual', tf.argmax(dis_class, 1))
        tf.summary.histogram('label_predicted', tf.argmax(dis_output_real_cls, 1))

    with tf.name_scope('summary_generator'):
        tf.summary.image('output', preproc.unapply(gen_output), max_outputs=32)
        tf.summary.scalar('loss/cls', gen_loss_cls)
        tf.summary.scalar('loss/dis', gen_loss_dis)
        tf.summary.scalar('loss', gen_loss)
        tf.summary.scalar('fooling_rate', gen_fooling)
        
    with tf.name_scope('summary_balance'):
        tf.summary.scalar('discriminator', log_step_dis)
        tf.summary.scalar('classifier', log_step_cls)
        tf.summary.scalar('generator', log_step_gen)

    # Summary operations:
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summary_dis = tf.summary.merge([v for v in summaries if "summary_discriminator/" in v.name])
    summary_cls = tf.summary.merge([v for v in summaries if "summary_classifier/" in v.name])
    summary_gen = tf.summary.merge([v for v in summaries if "summary_generator/" in v.name])
    summary_bal = tf.summary.merge([v for v in summaries if "summary_balance/" in v.name])

    increment_global_step = tf.assign_add(global_step, 1, name="increment_global_step")

    #
    # Training
    #
    if args.split == "train":
        logger = logging.getLogger("train")
        data = support.TrainData(args, preproc)

        sv = tf.train.Supervisor(logdir=config.get_filename(".", args), global_step=global_step, summary_op=None, save_model_secs=args.log_interval)
        with sv.managed_session() as sess:
            # Set up tensorboard logging:
            logwriter = tf.summary.FileWriter(config.get_filename(".", args), sess.graph)

            batch = sess.run(global_step)
            logwriter.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=batch)
            logger.info("Starting training from batch {}. Saving model every {}s.".format(batch, args.log_interval))

            # Format the score printing
            while not sv.should_stop():
                logger.debug('Step {} of {}.'.format(batch, args.batches))

                # This training proceeds in two phases: (1) discriminator, then (2) generator.
                # First, we train the discriminator for `step_dis` number of steps. Because the .trainable flag was True when 
                # `dis_model` was compiled, the weights of the discriminator will be updated. The discriminator is trained to
                # distinguish between "fake"" (generated) and real images by running it on one step of each.
                
                step_dis = 0
                while True:
                    # Generate fake images, and train the model to predict them as fake. We keep track of the loss in predicting
                    # Use real images (but not labels), and train the model to predict them as real. We perform both these at the
                    # same time so we can capture the summary op.
                    (_, loss_fake, fake_true_neg,), (_, loss_real, real_true_pos), summ_dis = sess.run(
                        ((train_dis_fake, dis_loss_fake, train_dis_fake_true_neg),
                        (train_dis_real, dis_loss_real, train_dis_real_true_pos), summary_dis), feed_dict={
                        gen_input_seed: next(data.rand_vec),
                        gen_input_class: next(data.rand_label_vec),
                        dis_label_fake: next(data.label_dis_fake),
                        dis_input: next(data.unlabelled)[0],
                        dis_label_real: next(data.label_dis_real)
                    })
                    step_dis += 1
                    logwriter.add_summary(summ_dis, global_step=batch)

                    if args.hyperparam.discriminator_halt(batch, step_dis, 
                        {"fake_loss": loss_fake, "fake_true_neg": fake_true_neg,
                        "real_loss": loss_real, "real_true_pos": real_true_pos}):
                        break
                
                # Train classifier
                step_cls = 0
                while True:
                    data_x, data_y = next(data.labelled)
                    _, loss_label, accuracy, summ_cls = sess.run(
                        [train_cls, cls_loss, cls_accuracy, summary_cls], feed_dict={
                        dis_input: data_x,
                        dis_class: data_y,
                        dis_label_real: next(data.label_dis_real)
                    })
                    step_cls += 1
                    logwriter.add_summary(summ_cls, global_step=batch)
                    
                    if args.hyperparam.classifier_halt(batch, step_cls, {"cls_loss": loss_label, "cls_accuracy": accuracy}):
                        break
                

                # Second, we train the generator for `step_gen` number of steps. The generator weights are the only weights 
                # updated in this step. We train "generator" so that "discriminatsor(generator(random)) == real". 
                # Specifically, we compose `dis_model` onto `gen_model`, and train the combined model so that given a random
                # vector, it classifies images as real.
                step_gen = 0
                while True:
                    _, loss, fooling_rate, summ_gen = sess.run(
                        [train_gen, gen_loss, gen_fooling, summary_gen], feed_dict={
                        gen_input_seed: next(data.rand_vec),
                        gen_input_class: next(data.rand_label_vec),
                        dis_label_real: next(data.label_gen_real)
                    })
                    step_gen += 1
                    logwriter.add_summary(summ_gen, global_step=batch)
                    
                    if args.hyperparam.generator_halt(batch, step_gen, {"gen_loss": loss, "gen_fooling": fooling_rate}):
                        break

                #
                # That is the entire training algorithm.
                #
                batch, summ_bal, _ = sess.run((increment_global_step, summary_bal, (log_step_dis_assign, log_step_gen_assign, log_step_cls_assign)),feed_dict={log_step_cls_val: step_cls, log_step_dis_val: step_dis, log_step_gen_val: step_gen})
                logwriter.add_summary(summ_bal, global_step=batch)

    #
    # Testing
    #
    elif args.split == "test":
        logger = logging.getLogger("test")
        data = support.TestData(args, preproc)
        
        logger.info("Starting tests.")
        
        num = 0
        acc = 0.0
        k = np.zeros((args.hyperparam.NUM_CLASSES, args.hyperparam.NUM_CLASSES))

        sv = tf.train.Supervisor(logdir=config.get_filename(".", args), global_step=global_step, summary_op=None, save_model_secs=0)
        with sv.managed_session() as sess:
            # Load weights
            for i, d in enumerate(data.labelled):
                data_x, data_y = d
                
                v = sess.run(dis_output_real_cls, feed_dict={dis_input: data_x})
                # Update the current accuracy score
                
                num += v.shape[0]
                vp = np.argmax(v, axis=1)
                vq = np.argmax(data_y, axis=1)

                acc += np.sum(vp == vq)
                for (x, y) in zip(vq, vp):
                        k[x, y] += 1.0

        # Rescale the confusion matrix    
        k = k/(np.sum(k, axis=1) + 1e-7)*100.0

        logger.info("Classifier Accuracy: {:.1f}%".format(acc/num*100))
        logger.info("Confusion Matrix [Actual Rows, Reported Columns] (% of Row):\n" + np.array_str(k, max_line_width=120, precision=1, suppress_small=True))


if __name__ == '__main__':
    logger.info("Started")
    try:
        main(support.argparser().parse_args())
    except:
        raise
    finally:
        logger.info("Halting")
