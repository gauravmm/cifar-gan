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
import png
import support
import tensorflow as tf
from keras import layers, models
from keras_diagram import ascii

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

    img_dim = args.generator.IMAGE_DIM


    #
    # Build Model
    #

    gen_input  = layers.Input(shape=args.generator.SEED_DIM)
    # dis_input  = layers.Input(shape=args.generator.IMAGE_DIM)
    
    # Input/output tensors:
    # Define and compile models
    # We compose the discriminator onto the generator to produce the combined model:
    gen_model = args.generator.generator(args.generator.SEED_DIM, args.generator.IMAGE_DIM)
    dis_model = args.discriminator.discriminator(args.generator.IMAGE_DIM)
    com_model = models.Model(inputs=[gen_input], outputs=[dis_model(gen_model(gen_input))], name='combined')

    gen_model.compile(optimizer=args.hyperparam.optimizer_gen, loss='binary_crossentropy')
    dis_model.compile(optimizer=args.hyperparam.optimizer_dis, loss='binary_crossentropy', metrics=support.METRICS)
    # The trainable flag only takes effect upon compilation. By setting it False here, we allow the discriminator weights
    # to be updated in the step where we learn dis_model directly (compiled above), but not in the step where we learn
    # gen_model (compiled below). This behaviour is important, see comments in the training loop for details.
    dis_model.trainable = False
    com_model.compile(optimizer=args.hyperparam.optimizer_gen, loss='binary_crossentropy', metrics=support.METRICS)

    logger.info("Compiled models.")
    logger.debug("Generative model structure:\n{}".format(ascii(gen_model)))
    logger.debug("Discriminative model structure:\n{}".format(ascii(dis_model)))


    #
    # Load weights
    #

    if args.resume:
        logger.info("Attempting to resume from saved checkpoints.")
        batch = support.resume(args, gen_model, dis_model)
        if batch:
            logger.info("Successfully resumed from batch {}".format(batch))
        else:
            logger.warn("Could not resume training.".format(batch))

    # Clear the files
    if batch is None:
        batch = support.clear(args)
        
        # Write CSV file headers
        with open(config.get_filename('csv', args), 'w') as csvfile:
            print("time, batch, " + ", ".join("{} {}".format(a, b) 
                                            for b in com_model.metrics_names
                                            for a in ["composite", "discriminator+real", "discriminator+fake"]),
                  file=csvfile)
        logger.debug("Wrote headers to CSV file {}".format(csvfile.name))


    assert batch is not None
    

    #
    # Load data
    #
    
    # NOTE: Keras requires Numpy input, so we cannot use Tensorflow's built-in data augmentation tools. We can instead use Keras' tools.
    data = support.Data(args)

    #
    # Training
    #

    logger.info("Starting training. Reporting metrics {} every {} steps.".format(", ".join(com_model.metrics_names), args.log_interval))

    # Loss value in the current log interval:
    intv_com_loss = np.zeros(shape=len(com_model.metrics_names))
    intv_dis_loss_real = np.zeros(shape=len(dis_model.metrics_names))
    intv_dis_loss_fake = np.copy(intv_dis_loss_real)

    # Format the score printing
    print_score = lambda scores: ", ".join("{}: {}".format(p, s) for p, s in zip(com_model.metrics_names, scores))

    for batch in range(batch, args.batches):
        logger.debug('Step {} of {}.'.format(batch, args.batches))

        # This training proceeds in two phases: (1) discriminator, then (2) generator.
        # First, we train the discriminator for `step_dis` number of steps. Because the .trainable flag was True when 
        # `dis_model` was compiled, the weights of the discriminator will be updated. The discriminator is trained to
        # distinguish between "fake"" (generated) and real images by running it on one step of each.
        for _ in range(args.hyperparam.discriminator_per_step):
            # Generate fake images, and train the model to predict them as fake. We keep track of the loss in predicting
            # fake images separately from real images.
            intv_dis_loss_fake += dis_model.train_on_batch(gen_model.predict(next(data.rand_vec)),
                                                           next(data.label_dis_fake))
            # Use real images, and train the model to predict them as real.
            intv_dis_loss_real += dis_model.train_on_batch(next(data.real),
                                                           next(data.label_dis_real))

        # Second, we train the generator for `step_gen` number of steps. Because the .trainable flag (for `dis_model`) 
        # was False when `com_model` was compiled, the discriminator weights are not updated. The generator weights are
        # the only weights updated in this step.
        # In this step, we train "generator" so that "discriminator(generator(random)) == real". Specifically, we compose
        # `dis_model` onto `gen_model`, and train the combined model so that given a random vector, it classifies images
        # as real.
        for _ in range(args.hyperparam.generator_per_step):
            intv_com_loss += com_model.train_on_batch(next(data.rand_vec), next(data.label_gen_real))

        # That is the entire training algorithm.
        # Produce output every interval:
        if not batch % args.log_interval and batch != 0:
            logger.info("Completed batch {}/{}".format(batch, args.batches))

            # Compute the average loss over this interval
            intv_com_loss      /= args.log_interval * args.hyperparam.generator_per_step
            intv_dis_loss_fake /= args.log_interval * args.hyperparam.discriminator_per_step
            intv_dis_loss_real /= args.log_interval * args.hyperparam.discriminator_per_step

            # Log a summary
            logger.info("Generator; {}.".format(print_score(intv_com_loss)))
            logger.info("Discriminator on real; {}.".format(print_score(intv_dis_loss_real)))
            logger.info("Discriminator on fake; {}.".format(print_score(intv_dis_loss_fake)))

            # Log to CSV
            with open(config.get_filename('csv', args), 'a') as csvfile:
                fmt_metric = lambda x: ", ".join(str(v) for v in x)
                print("{}, {}, {}, {}, {}".format(
                    datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                    batch,
                    fmt_metric(intv_com_loss),
                    fmt_metric(intv_dis_loss_real),
                    fmt_metric(intv_dis_loss_fake)), file=csvfile)
            
            # Zero out the running counters
            intv_com_loss[...]      = 0
            intv_dis_loss_fake[...] = 0
            intv_dis_loss_real[...] = 0

            # Write image
            img_fn = config.get_filename('image', args, batch)
            img_data = data.unapply(gen_model.predict(next(data.rand_vec)))
            png.from_array(support.arrange_images(img_data, args), 'RGB').save(img_fn)
            logger.debug("Saved sample images to {}.".format(img_fn))

            # Save weights
            gen_model.save_weights(config.get_filename('weight', args, 'gen', batch))
            dis_model.save_weights(config.get_filename('weight', args, 'dis', batch))
            logger.debug("Saved weights for batch {}.".format(batch))


if __name__ == '__main__':
    logger.info("Started")
    try:
        main(support.argparser().parse_args())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt while processing batch {}".format(batch))
    except:
        raise
    finally:
        logger.info("Halting")
