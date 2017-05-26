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
logfile = logging.FileHandler(filename=config.PATH['log'], mode='w')
logfile.setLevel(logging.DEBUG)
logfile.setFormatter(logging.Formatter('[%(asctime)s, %(levelname)s @%(name)s] %(message)s'))
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s @%(name)s] %(message)s', datefmt='%H:%M:%S'))
logging.basicConfig(level=logging.DEBUG, handlers=[logfile, console])
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logger = logging.getLogger()

def main(args):
    """
    Main class, does:
      (1) Model building
      (2) Loading
      (3) Training
    
    args contains the commandline arguments, and the classes specified by commandline argument.
    """

    logger.debug("Commandline Arguments: " + str(args))

    logger.info("Loaded dataset        : {}".format(args.data.__file__))
    logger.info("Loaded generator      : {}".format(args.generator.__file__))
    logger.info("Loaded discriminator  : {}".format(args.discriminator.__file__))
    logger.info("Loaded hyperparameters: {}".format(args.hyperparam.__file__))
    logger.info("Loaded preprocessors  : {}".format(", ".join(a.__file__ for a in args.preprocessor)))
    
    run(args)


_shape_str = lambda a: "(" + ", ".join("?" if b is None else str(b) for b in a) + ")"
def run(args):
    logger = logging.getLogger("build")

    logger.info("WGAN Mode {}".format("Enabled" if args.hyperparam.WGAN_ENABLE else "Disabled"))

    #
    # Build Model
    #
    
    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False, dtype=tf.int32)
    is_training = tf.placeholder(tf.bool, shape=None, name='global_is_training')

    gen_input_seed  = tf.placeholder(tf.float32, shape=[None] + list(args.hyperparam.SEED_DIM), name="input_gen_seed")
    gen_input_class  = tf.placeholder(tf.float32, shape=(None, args.hyperparam.NUM_CLASSES), name="input_gen_class")
    with tf.variable_scope('model_generator'):
        gen_output = args.generator.generator(gen_input_seed, is_training, gen_input_class, args.hyperparam.IMAGE_DIM)
        
        # Sanity checking output
        if not str(gen_output.get_shape()) == _shape_str([None] + list(args.hyperparam.IMAGE_DIM)):
            logger.error("Generator output size is incorrect! Expected: {}, actual: {}".format(
                    _shape_str([None] + list(args.hyperparam.IMAGE_DIM)), str(gen_output.get_shape())))

    dis_input      = tf.placeholder(tf.float32, shape=[None] + list(args.hyperparam.IMAGE_DIM), name="input_dis_image")
    dis_label_real = tf.placeholder(tf.float32, shape=[None], name="input_dis_label_real")
    dis_label_fake = tf.placeholder(tf.float32, shape=[None], name="input_dis_label_real")
    dis_class      = tf.placeholder(tf.float32, shape=(None, args.hyperparam.NUM_CLASSES), name="input_dis_class")
    with tf.variable_scope('model_discriminator'):
        # Make sure that the generator and real images are the same size:
        assert str(gen_output.get_shape()) == str(dis_input.get_shape())
        with tf.variable_scope('real'):
            dis_output_real_dis, dis_output_real_cls = args.discriminator.discriminator(dis_input, is_training, args.hyperparam.NUM_CLASSES)
        with tf.variable_scope('fake'):
            dis_output_fake_dis, dis_output_fake_cls = args.discriminator.discriminator(gen_output, is_training, args.hyperparam.NUM_CLASSES)

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

    # The TensorFlow / operation automatically coerces the output type to a float. See [here](https://www.tensorflow.org/versions/master/api_docs/python/tf/divide).
    func_loss = tf.nn.sigmoid_cross_entropy_with_logits
    #func_loss = lambda **kwargs: tf.nn.l2_loss(kwargs["labels"] - kwargs["logits"])
    count_fraction = lambda x: tf.reduce_mean(tf.cast(x, tf.float32))
    with tf.name_scope('metrics'):
        # Discriminator losses
        with tf.name_scope('dis_fake'):
            with tf.name_scope('loss'):
                dis_loss_fake = tf.reduce_mean(func_loss(labels=dis_label_fake, logits=dis_output_fake_dis))
            with tf.name_scope('true_neg'):
                train_dis_fake_true_neg = count_fraction(tf.greater_equal(dis_output_fake_dis, 0.5))
        with tf.name_scope('dis_real'):
            with tf.name_scope('loss'):
                dis_loss_real = tf.reduce_mean(func_loss(labels=dis_label_real, logits=dis_output_real_dis))
            with tf.name_scope('true_pos'):
                train_dis_real_true_pos = count_fraction(tf.less(dis_output_real_dis, 0.5))

        dis_loss = dis_loss_real + dis_loss_fake

        # Classifier loss
        with tf.name_scope('cls'):
            with tf.name_scope('dis'):
                cls_loss_dis  = tf.reduce_mean(func_loss(labels=dis_label_real, logits=dis_output_real_dis))
            with tf.name_scope('cls'):
                cls_loss_cls  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=dis_class, logits=dis_output_real_cls))
            cls_loss = args.hyperparam.loss_weights_classifier["discriminator"] * cls_loss_dis \
                    + args.hyperparam.loss_weights_classifier["classifier"]    * cls_loss_cls
            
            with tf.name_scope('acc'):
                cls_accuracy = count_fraction(tf.equal(tf.argmax(dis_class, axis=1), tf.argmax(dis_output_real_cls, axis=1)))

        # Generator loss
        with tf.name_scope('gen'):
            with tf.name_scope('dis'):
                gen_loss_dis  = tf.reduce_mean(func_loss(labels=dis_label_real, logits=dis_output_fake_dis))
            with tf.name_scope('cls'):
                gen_loss_cls  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gen_input_class, logits=dis_output_fake_cls))
            gen_loss = args.hyperparam.loss_weights_generator["discriminator"] * gen_loss_dis \
                     + args.hyperparam.loss_weights_generator["classifier"]    * gen_loss_cls
            
            with tf.name_scope("fooling_rate"):
                gen_fooling = count_fraction(tf.less(dis_output_fake_dis, 0.5))

        # WGAN loss:
        with tf.name_scope('wgan'):
            with tf.name_scope('gen'):
                gen_loss_wgan = tf.reduce_mean(dis_output_fake_dis)
            with tf.name_scope('dis'):
                dis_loss_wgan = tf.reduce_mean(dis_output_real_dis) - tf.reduce_mean(dis_output_fake_dis)
            

        logger.info("Model constructed.")

    # Preprocessor
    preproc = support.Preprocessor(args)

    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    discriminator_variables = [v for v in variables if 'model_discriminator/' in v.name]
    logger.debug("Discriminator + Classifier Variables::\n\t{}".format(",\n\t".join(v.name for v in discriminator_variables)))
    generator_variables     = [v for v in variables if 'model_generator/' in v.name]
    logger.debug("Generator Variables:\n\t{}".format(",\n\t".join(v.name for v in generator_variables)))
    if args.hyperparam.WGAN_ENABLE:
        wgan_clipped_variables  =  [v for v in discriminator_variables if "model_discriminator/classifier/" not in v.name]
        logger.debug("WGAN Clipped Variables:\n\t{}".format(",\n\t".join(v.name for v in wgan_clipped_variables)))

    # Train ops
    with tf.name_scope('train_ops'):
        # These operations may be used in the future. For now, a combined train_dis handles the discriminator training
        # train_dis_fake = args.hyperparam.optimizer_dis. \
        #                    minimize(dis_loss_fake, var_list=discriminator_variables)
        # train_dis_real = args.hyperparam.optimizer_dis. \
        #                    minimize(dis_loss_real, var_list=discriminator_variables)

        train_dis = args.hyperparam.optimizer_dis. \
                            minimize(dis_loss, var_list=discriminator_variables)
        train_cls = args.hyperparam.optimizer_cls. \
                            minimize(cls_loss, var_list=discriminator_variables)
        train_gen = args.hyperparam.optimizer_gen. \
                            minimize(gen_loss, var_list=generator_variables)
        
        if args.hyperparam.WGAN_ENABLE:
            logger.info("Building WGAN optimizer.")
            with tf.name_scope('wgan'):
                train_dis = args.hyperparam.optimizer_dis. \
                                    minimize(dis_loss_wgan, var_list=discriminator_variables)
                train_gen = args.hyperparam.optimizer_dis. \
                                    minimize(gen_loss_wgan, var_list=generator_variables)
                logger.warn("There is no classifier implementation for WGAN. Ensure that `ENABLE_TRAINING_CLS` is False in your hyperparameter definitions.")


    # Clipping operations for WGAN support:
    if args.hyperparam.WGAN_ENABLE:
        with tf.name_scope('wgan_ops'):
            wgan_dis_clip = tf.group(*[v.assign(tf.clip_by_value(v, -1*args.hyperparam.WGAN_DIS_CLIP, args.hyperparam.WGAN_DIS_CLIP)) for v in wgan_clipped_variables])
        

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

            if args.hyperparam.SUMMARIZE_MORE:
                with tf.name_scope('debug'):        
                    tf.summary.histogram('label', dis_label_fake)
                    tf.summary.histogram('crossentropy', func_loss(labels=dis_label_fake, logits=dis_output_fake_dis))
                    tf.summary.histogram('pre_generator_output', gen_output)
                    tf.summary.image('generator_output', preproc.unapply(gen_output), max_outputs=32)
                    tf.summary.image('real_input', preproc.unapply(dis_input), max_outputs=32)

        with tf.name_scope('real'):
            tf.summary.scalar('loss', dis_loss_real)
            tf.summary.scalar('true_pos', train_dis_real_true_pos)
            tf.summary.histogram('dis', dis_output_real_dis)

            if args.hyperparam.SUMMARIZE_MORE:
                with tf.name_scope('debug'):        
                    tf.summary.histogram('label', dis_label_real)
                    tf.summary.histogram('crossentropy', func_loss(labels=dis_label_real, logits=dis_output_real_dis))
        
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

    if not args.hyperparam.ENABLE_TRAINING_DIS:
        logger.warn("Training the discriminator is disabled! If this is not intentional, set `ENABLE_TRAINING_DIS = True` in your hyperparameter definition.")
    if not args.hyperparam.ENABLE_TRAINING_CLS:
        logger.warn("Training the classifier is disabled! If this is not intentional, set `ENABLE_TRAINING_CLS = True` in your hyperparameter definition.")
    if not args.hyperparam.ENABLE_TRAINING_GEN:
        logger.warn("Training the generator is disabled! If this is not intentional, set `ENABLE_TRAINING_GEN = True` in your hyperparameter definition.")

    logger.warn(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

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
            logger.info("Starting training from batch {} to {}. Saving model every {}s.".format(batch, args.batches, args.log_interval))

            # Format the score printing
            while not sv.should_stop():
                if batch >= args.batches:
                    logger.info("Step {}; limit reached, Saving and halting...".format(batch))
                    sv.saver.save(sess, config.get_filename("model.ckpt", args), global_step=batch)
                    sv.stop()
                    break

                if batch % 100 == 0:
                    logger.debug('Step {} of {}.'.format(batch, args.batches))

                # This training proceeds in two phases: (1) discriminator, then (2) generator.
                # First, we train the discriminator for `step_dis` number of steps. Because the .trainable flag was True when 
                # `dis_model` was compiled, the weights of the discriminator will be updated. The discriminator is trained to
                # distinguish between "fake"" (generated) and real images by running it on one step of each.                
                step_dis = 0
                if args.hyperparam.ENABLE_TRAINING_DIS:
                    while True:
                        # Generate fake images, and train the model to predict them as fake. We keep track of the loss in predicting
                        # Use real images (but not labels), and train the model to predict them as real. We perform both these at the
                        # same time so we can capture the summary op.
                        _, (loss_fake, fake_true_neg, loss_real, real_true_pos), summ_dis = sess.run(
                            [train_dis, (dis_loss_fake, train_dis_fake_true_neg, dis_loss_real, train_dis_real_true_pos), summary_dis], feed_dict={
                            gen_input_seed: next(data.rand_vec),
                            gen_input_class: next(data.rand_label_vec),
                            dis_label_fake: next(data.label_dis_fake),
                            dis_input: next(data.unlabelled)[0],
                            dis_label_real: next(data.label_dis_real),
                            is_training: True
                        })

                        # WGANs require weight clipping.
                        if args.hyperparam.WGAN_ENABLE:
                            sess.run(wgan_dis_clip)

                        step_dis += 1
                        logwriter.add_summary(summ_dis, global_step=batch)

                        if args.hyperparam.discriminator_halt(batch, step_dis, 
                            {"fake_loss": loss_fake, "fake_true_neg": fake_true_neg,
                            "real_loss": loss_real, "real_true_pos": real_true_pos}):
                            break
                
                # Second, we train the classifier
                step_cls = 0
                if args.hyperparam.ENABLE_TRAINING_CLS:
                    if args.hyperparam.WGAN_ENABLE:
                        logger.error("Attempting to train a classifier with WGANs. I told you this wouldn\'t work, didn't I?")
                        logger.error("How to resolve:")
                        logger.error("    1) Set `ENABLE_TRAINING_CLS = False` in your hyperparam file to disable training the classifier.")
                        logger.error("    2) Switch to a hyperparam file configured for normal GANs.")
                        logger.error("    3) Figure out the theoretical challenges of training a classifier with Wasserstein distances, delve into the source code of this program, implement it, submit a pull request against git@github.com:gauravmm/cifar-gan.git:development, and write a paper about the process.")
                        logger.error("    4) Wait for Gaurav to do (3).")
                        
                        raise RuntimeError('Training classifier with WGAN is unsupported.')
                    while True:
                        data_x, data_y = next(data.labelled)
                        _, loss_label, accuracy, summ_cls = sess.run(
                            [train_cls, cls_loss, cls_accuracy, summary_cls], feed_dict={
                            dis_input: data_x,
                            dis_class: data_y,
                            dis_label_real: next(data.label_dis_real),
                            is_training: True
                        })
                        step_cls += 1
                        logwriter.add_summary(summ_cls, global_step=batch)
                        
                        if args.hyperparam.classifier_halt(batch, step_cls, {"cls_loss": loss_label, "cls_accuracy": accuracy}):
                            break
                

                # Finally, we train the generator for `step_gen` number of steps. The generator weights are the only weights 
                # updated in this step. We train "generator" so that "discriminatsor(generator(random)) == real". 
                # Specifically, we compose `dis_model` onto `gen_model`, and train the combined model so that given a random
                # vector, it classifies images as real.
                step_gen = 0
                if args.hyperparam.ENABLE_TRAINING_GEN:
                    while True:
                        _, loss, fooling_rate, summ_gen = sess.run(
                            [train_gen, gen_loss, gen_fooling, summary_gen], feed_dict={
                            gen_input_seed: next(data.rand_vec),
                            gen_input_class: next(data.rand_label_vec),
                            dis_label_real: next(data.label_gen_real),
                            is_training: True
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
                
                v = sess.run(dis_output_real_cls, feed_dict={dis_input: data_x, is_training: False})
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
