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
from keras import layers, models
from keras.utils import plot_model

import config
import support

#
# Init
#

np.random.seed(54183)

# Logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
support._make_path(config.PATH['logs'])
logging.basicConfig(filename=os.path.join(config.PATH['logs'], 'adversarial.log'), level=logging.DEBUG, format='[%(asctime)s, %(levelname)s] %(message)s')
# Logger
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s] %(message)s', datefmt='%H:%M:%S'))
logging.getLogger().addHandler(console)
logger = logging.getLogger()

global batch
batch = None
def main(args):
    """
    Main class, does:
      (1) Model building
      (2) Loading
      (3) Training
    
    args contains the commandline arguments, and the classes specified by commandline argument.
    """
    global batch

    logger.info("Loaded dataset        : {}".format(args.data.__file__))
    logger.info("Loaded generator      : {}".format(args.generator.__file__))
    logger.info("Loaded discriminator  : {}".format(args.discriminator.__file__))
    logger.info("Loaded hyperparameters: {}".format(args.hyperparam.__file__))
    logger.info("Loaded preprocessors  : {}".format(", ".join(a.__file__ for a in args.preprocessor)))

    #
    # Build Model
    #

    # Input/output tensors:
    gen_input  = layers.Input(shape=args.hyperparam.SEED_DIM, name="input_gen_seed")
    gen_label_input  = layers.Input(shape=(args.hyperparam.NUM_CLASSES,), name="input_gen_class")
    dis_input  = layers.Input(shape=args.hyperparam.IMAGE_DIM, name='input_dis')
    
    dis_output           = args.discriminator.discriminator(dis_input, args.hyperparam.NUM_CLASSES)
    dis_model_labelled   = models.Model(inputs=dis_input, outputs=dis_output, name='model_discriminator_labelled')
    dis_model_unlabelled = models.Model(inputs=dis_input, outputs=dis_output, name='model_discriminator_unlabelled')
    
    gen_output = args.generator.generator(gen_input, gen_label_input, args.hyperparam.IMAGE_DIM)
    gen_model = models.Model(inputs=[gen_input, gen_label_input], outputs=gen_output, name="model_generator")
    gen_model.compile(optimizer=args.hyperparam.optimizer_gen, loss='binary_crossentropy')
    # We compose the discriminator onto the generator to produce the combined model:

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
    
    for v, f in [(com_model, "com_model.png"), (dis_model_labelled, "dis_model_labelled.png"), (dis_model_unlabelled, "dis_model_unlabelled.png"), (gen_model, "gen_model.png")]:
        plot_model(v, show_shapes=True, to_file=os.path.join(config.PATH['logs'], f))
    logger.debug("Model structures written to {}".format(config.PATH['logs']))

    metrics_names = support.get_metric_names(com_model, dis_model_labelled, dis_model_unlabelled, gen_model)

    #
    # Load weights
    #

    if args.split == "test":
        logger.info("Attempting to load last checkpoint for testing.")
        batch = support.resume(args, gen_model, dis_model_labelled)
        if batch:
            logger.info("Successfully loaded batch {}".format(batch))
        else:
            logger.warn("Could not load latest checkpoint for testing. Exiting...")
            return
    elif args.resume:
        logger.info("Attempting to resume from saved checkpoints.")
        batch = support.resume(args, gen_model, dis_model_labelled)
        if batch:
            logger.info("Successfully resumed from batch {}".format(batch))
        else:
            logger.warn("Could not resume training.")

    # Clear the files
    if batch is None:
        assert args.split == "train"
        batch = support.clear(args)
        
        # Write CSV file headers
        with open(config.get_filename('csv', args), 'w') as csvfile:
            print("time, batch, " + ", ".join("{} {}".format(a, b) 
                                            for a in ["composite", "discriminator+real", "discriminator+fake"]
                                            for b in metrics_names) +
                  ", discriminator_steps, generator_steps",
                  file=csvfile)
        logger.debug("Wrote headers to CSV file {}".format(csvfile.name))


    assert batch is not None

    if args.split == "train":
        try:
            train(args, metrics_names, (dis_model_unlabelled, dis_model_labelled, gen_model, com_model))
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt while training on batch {}.".format(batch))
    elif args.split == "test":
        try:
            test(args, metrics_names, (dis_model_unlabelled, dis_model_labelled, gen_model, com_model))
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt while testing.")
    else:
        assert not "This state should not be reachable; the argparser should catch this case."

def save_sample_images(args, data, batch, gen_model):
    img_fn = config.get_filename('image', args, batch)
    img_data = data.unapply(gen_model.predict([next(data.rand_vec), next(data.rand_label_vec)]))
    png.from_array(support.arrange_images(img_data, args), 'RGB').save(img_fn)
    logger.debug("Saved sample images to {}.".format(img_fn))


#
# Training
#

def test(args, metrics_names, models):
    dis_model_unlabelled, dis_model_labelled, gen_model, com_model = models

    metric_wrap = lambda x: {k:v for k, v in zip(metrics_names, x)}
    data = support.TestData(args, "test")

    logger.info("Starting tests.")
    metrics = np.zeros(shape=len(metrics_names))
    i = 0
    for batch, d in enumerate(data.labelled):
        data_x, data_y = d
        m = dis_model_labelled.test_on_batch(data_x, [next(data.label_dis_real), data_y])
        metrics += m
        i += 1

    metrics /= i
    m = metric_wrap(metrics)
    logger.debug(m)
    logger.info("Classifier Accuracy: {:.1f}%".format(m['classifier_acc']*100))
    logger.info("Discriminator Accuracy: {:.1f}%".format(m['discriminator_label_real']*100))


def train(args, metrics_names, models):
    dis_model_unlabelled, dis_model_labelled, gen_model, com_model = models
    global batch

    data = support.TrainData(args)

    logger.info("Starting training. Reporting metrics {} every {} steps.".format(", ".join(metrics_names), args.log_interval))

    # Loss value in the current log interval:
    intv_com_loss = np.zeros(shape=len(metrics_names))
    intv_dis_loss_real = np.zeros(shape=len(metrics_names))
    intv_dis_loss_fake = np.copy(intv_dis_loss_real)
    intv_com_count = 0
    intv_dis_count = 0

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
        
        # Train with labels
        step_dis_label = 0
        while not args.hyperparam.classifier_halt(batch, step_dis_label):
            data_x, data_y = next(data.labelled)
            loss_label = dis_model_labelled.train_on_batch(
                data_x,
                [ next(data.label_dis_real), data_y ])
            step_dis_loss_real += loss_real
            step_dis_label += 1


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
                fmt_metric(step_com_loss / step_com),
                fmt_metric(step_dis_loss_real / step_dis),
                fmt_metric(step_dis_loss_fake / step_dis),
                step_dis,
                step_com), file=csvfile)
        
        intv_dis_count += step_dis
        intv_dis_loss_fake += step_dis_loss_fake
        intv_dis_loss_real += step_dis_loss_real
        intv_com_loss += step_com_loss
        intv_com_count += step_com
        
        # Produce output every args.log_interval. We increment batch number because we have just finished the batch.
        batch += 1
        if not batch % args.log_interval:
            logger.info("Completed batch {}/{}".format(batch, args.batches))

            # Compute the average loss over this interval
            intv_com_loss      /= intv_com_count
            intv_dis_loss_fake /= intv_dis_count
            intv_dis_loss_real /= intv_dis_count

            # Log a summary
            logger.info("Generator; {}.".format(print_score(intv_com_loss)))
            logger.info("Discriminator on real; {}.".format(print_score(intv_dis_loss_real)))
            logger.info("Discriminator on fake; {}.".format(print_score(intv_dis_loss_fake)))

            # Zero out the running counters
            intv_com_loss[...]      = 0
            intv_dis_loss_fake[...] = 0
            intv_dis_loss_real[...] = 0
            intv_com_count = 0
            intv_dis_count = 0

            # Write image
            save_sample_images(args, data, batch, gen_model)
            
            # Save weights
            gen_model.save_weights(config.get_filename('weight', args, 'gen', batch))
            dis_model_labelled.save_weights(config.get_filename('weight', args, 'dis', batch))
            logger.debug("Saved weights for batch {}.".format(batch))


if __name__ == '__main__':
    logger.info("Started")
    try:
        main(support.argparser().parse_args())
    except:
        raise
    finally:
        logger.info("Halting")
