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

    dis_input  = tf.placeholder(tf.float32, shape=[None] + list(args.hyperparam.IMAGE_DIM), name="input_dis")
    with tf.variable_scope('model_discriminator') as disc_scope:
        dis_output_real_dis, dis_output_real_cls = args.discriminator.discriminator(dis_input, args.hyperparam.NUM_CLASSES)
        disc_scope.reuse_variables()
        dis_output_fake_dis, dis_output_fake_cls = args.discriminator.discriminator(gen_output, args.hyperparam.NUM_CLASSES)

    logger.info("Model constructed.")
    data = support.TrainData(args)

    sv = tf.train.Supervisor(logdir=config.get_filename(".", args), global_step=global_step, save_summaries_secs=60, save_model_secs=600)
    with sv.managed_session() as sess:

        # Prepare summaries
        tf.summary.image('summary/generator', data.unapply(), max_outputs=8)
        """
        # Prepare summaries:
        
        tf.summary.scalar('summary/discriminator_loss', discriminator_loss)
        tf.summary.scalar('summary/generator_loss', generator_loss)

        gen_model.compile(optimizer=args.hyperparam.optimizer_gen, loss='binary_crossentropy')

        
        _dis_model_compile = {
                'optimizer': args.hyperparam.optimizer_dis,
                'loss': [args.hyperparam.loss_func['discriminator'], args.hyperparam.loss_func['classifier']],
                'metrics': support.METRICS
            }
        dis_model_labelled.compile(
            loss_weights=[args.hyperparam.loss_weights['discriminator'], args.hyperparam.loss_weights['classifier']],
            **_dis_model_compile)
        dis_model_unlabelled.compile(loss_weights=[1, 0], **_dis_model_compile)
        
        com_model = models.Model(inputs=[gen_input, gen_label_input], outputs=dis_model_unlabelled(gen_model([gen_input, gen_label_input])), name='model_combined')
        # The trainable flag only takes effect upon compilation. By setting it False here, we allow the discriminator weights
        # to be updated in the step where we learn dis_model directly (compiled above), but not in the step where we learn
        # gen_model (compiled below). This behaviour is important, see comments in the training loop for details.
        dis_model_unlabelled.trainable = False
        com_model.compile(optimizer=args.hyperparam.optimizer_gen,
                        loss=[args.hyperparam.loss_func['discriminator'], args.hyperparam.loss_func['classifier']],
                        loss_weights=[1, 0],
                        metrics=support.METRICS)

        logger.info("Compiled models.")
        
        metrics_names = support.get_metric_names(com_model, dis_model_labelled, dis_model_unlabelled, gen_model)
        """


        logger.info("Starting training. Reporting metrics {} every {} steps.".format(", ".join(metrics_names), args.log_interval))

        # Loss value in the current log interval:
        intv_com_loss = np.zeros(shape=len(metrics_names))
        intv_dis_loss_real = np.zeros(shape=len(metrics_names))
        intv_dis_loss_fake = np.zeros(shape=len(metrics_names))
        intv_cls_loss = np.zeros(shape=len(metrics_names))
        intv_com_count = 0
        intv_dis_count = 0
        intv_cls_count = 0

        # Format the score printing
        zero_loss = lambda: np.asarray([0. for _ in metrics_names], dtype=np.float16)
        print_score = lambda scores: ", ".join("{}: {}".format(p, s) for p, s in zip(metrics_names, scores))
        metric_wrap = lambda x: {k:v for k, v in zip(metrics_names, x)}
        for batch in range(batch, args.batches):
            logger.debug('Step {} of {}.'.format(batch, args.batches))

            # This training proceeds in two phases: (1) discriminator, then (2) generator.
            # First, we train the discriminator for `step_dis` number of steps. Because the .trainable flag was True when 
            # `dis_model` was compiled, the weights of the discriminator will be updated. The discriminator is trained to
            # distinguish between "fake"" (generated) and real images by running it on one step of each.
            
            step_dis = 0
            step_dis_loss_fake = zero_loss()
            step_dis_loss_real = zero_loss()
            while True:
                # Generate fake images, and train the model to predict them as fake. We keep track of the loss in predicting
                # fake images separately from real images.
                fake_class = next(data.rand_label_vec)
                loss_fake = dis_model_unlabelled.train_on_batch(
                    gen_model.predict([next(data.rand_vec), fake_class]),
                    [next(data.label_dis_fake), fake_class])
                step_dis_loss_fake += loss_fake
                
                # Use real images (but not labels), and train the model to predict them as real.
                data_x, _ = next(data.unlabelled)
                loss_real = dis_model_unlabelled.train_on_batch(
                    data_x, 
                    [next(data.label_dis_real), fake_class])
                step_dis_loss_real += loss_real

                step_dis += 1
                if args.hyperparam.discriminator_halt(batch, step_dis, metric_wrap(loss_fake), metric_wrap(loss_real)):
                    break
            
            # Train classifier
            step_cls = 0
            step_cls_loss = zero_loss()
            while not args.hyperparam.classifier_halt(batch, step_cls):
                data_x, data_y = next(data.labelled)
                loss_label = dis_model_labelled.train_on_batch(
                    data_x,
                    [ next(data.label_dis_real), data_y ])
                step_cls_loss += loss_label
                step_cls += 1


            # Second, we train the generator for `step_gen` number of steps. Because the .trainable flag (for `dis_model`) 
            # was False when `com_model` was compiled, the discriminator weights are not updated. The generator weights are
            # the only weights updated in this step.
            # In this step, we train "generator" so that "discriminator(generator(random)) == real". Specifically, we compose
            # `dis_model` onto `gen_model`, and train the combined model so that given a random vector, it classifies images
            # as real.
            step_com = 0
            step_com_loss = zero_loss()
            while True:
                fake_class = next(data.rand_label_vec)
                loss = com_model.train_on_batch(
                    [ next(data.rand_vec), fake_class ],
                    [ next(data.label_gen_real), fake_class ])
                step_com_loss += loss
                
                step_com += 1
                if args.hyperparam.generator_halt(batch, step_com, metric_wrap(loss)):
                    break

            #
            # That is the entire training algorithm.
            #

            # Log to CSV
            with open(config.get_filename('csv', args), 'a') as csvfile:
                fmt_metric = lambda x: ", ".join(str(v) for v in x)
                print("{}, {}, {}, {}, {}, {}, {}".format(
                    int(time.time()),
                    batch,
                    fmt_metric(step_dis_loss_real / step_dis),
                    fmt_metric(step_dis_loss_fake / step_dis),
                    fmt_metric(step_cls_loss / step_cls),
                    fmt_metric(step_com_loss / step_com),
                    step_dis,
                    step_com), file=csvfile)
            
            intv_dis_loss_fake += step_dis_loss_fake
            intv_dis_loss_real += step_dis_loss_real
            intv_cls_loss += step_cls_loss
            intv_com_loss += step_com_loss

            intv_dis_count += step_dis
            intv_cls_count += step_cls
            intv_com_count += step_com
            
            # Produce output every args.log_interval. We increment batch number because we have just finished the batch.
            batch += 1
            if not batch % args.log_interval:
                logger.info("Completed batch {}/{}".format(batch, args.batches))

                # Log a summary of the average loss this interval
                logger.info("Discriminator on real; {}.".format(print_score(intv_dis_loss_real / intv_dis_count)))
                logger.info("Discriminator on fake; {}.".format(print_score(intv_dis_loss_fake / intv_dis_count)))
                logger.info("Classifier; {}.".format(print_score(intv_cls_loss / intv_cls_count)))
                logger.info("Generator; {}.".format(print_score(intv_com_loss / intv_com_count)))

                # Zero out the running counters
                intv_dis_loss_fake[...] = 0
                intv_dis_loss_real[...] = 0
                intv_cls_loss[...]      = 0
                intv_com_loss[...]      = 0

                intv_dis_count = 0
                intv_cls_count = 0
                intv_com_count = 0

                # Write image
                save_sample_images(args, data, batch, gen_model)
                
                # Save weights
                gen_model.save_weights(config.get_filename('weight', args, 'gen', batch))
                dis_model_labelled.save_weights(config.get_filename('weight', args, 'dis', batch))
                logger.debug("Saved weights for batch {}.".format(batch))




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