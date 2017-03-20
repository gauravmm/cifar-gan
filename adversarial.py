#!/bin/python3

"""
Main CIFAR-GAN file.
Handles the loading of data from ./data, models from ./models, training, and testing.

Modified from TensorFlow-Slim examples and https://github.com/wayaai/GAN-Sandbox/
"""

import datetime
import itertools
import logging
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
logging.basicConfig(filename=os.path.join(config.PATH['logs'], 'adversarial.log'), level=logging.DEBUG, format='[%(asctime)s, %(levelname)s] %(message)s')
# Logger
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s] %(message)s', datefmt='%H:%M:%S'))
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
    logger.info("Loaded dataset      : \t{}".format(args.data.__file__))
    logger.info("Loaded generator    : \t{}".format(args.generator.__file__))
    logger.info("Loaded discriminator: \t{}".format(args.discriminator.__file__))
    logger.info("Loaded hyperparameters: \t{}".format(args.hyperparam.__file__))
    logger.info("Loaded preprocessors: \t{}".format(", ".join(a.__file__ for a in args.preprocessor)))

    img_dim = args.generator.IMAGE_DIM


    #
    # Build Model
    #

    # TODO: Update calls to Keras 2 API to remove warnings. We may have to restructure the code accordingly
    # See use of new API here: https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

    # Input/output tensors:
    gen_input  = layers.Input(shape=args.generator.SEED_DIM)
    gen_output = args.generator.generator(gen_input, img_dim)
    gen_model = models.Model(input=gen_input, output=gen_output, name='generator')
    
    dis_input  = layers.Input(shape=args.generator.IMAGE_DIM)
    dis_output = args.discriminator.discriminator(dis_input, img_dim)
    dis_model = models.Model(input=dis_input, output=dis_output, name='discriminator')
    
    # Define and compile models
    # We compose the discriminator onto the generator to produce the combined model:
    com_model = models.Model(input=gen_input, output=dis_model(gen_model(gen_input)), name='combined')

    gen_model.compile(optimizer=args.hyperparam.optimizer, loss='binary_crossentropy')
    dis_model.compile(optimizer=args.hyperparam.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # The trainable flag only takes effect upon compilation. By setting it False here, we allow the discriminator weights
    # to be updated in the step where we learn dis_model directly (compiled above), but not in the step where we learn
    # gen_model (compiled below). This behaviour is important, see comments in the training loop for details.
    dis_model.trainable = False 
    com_model.compile(optimizer=args.hyperparam.optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    logger.info("Compiled models.")
    logger.debug("Generative model structure:\n{}".format(ascii(gen_model)))
    logger.debug("Discriminative model structure:\n{}".format(ascii(dis_model)))


    #
    # Load weights
    #

    batch = None
    if args.resume:
        batch = support.resume(args, gen_model, dis_model)
        if batch:
            logger.info("Resuming from batch {}".format(batch))
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

    for batch in range(batch, args.hyperparam.halt_batches):
        logger.debug('Step {} of {}.'.format(batch, args.hyperparam.halt_batches))

        # This training proceeds in two phases: (1) discriminator, then (2) generator.
        # First, we train the discriminator for `step_dis` number of steps. Because the .trainable flag was True when 
        # `dis_model` was compiled, the weights of the discriminator will be updated. The discriminator is trained to
        # distinguish between "fake"" (generated) and real images by running it on one step of each.
        for _ in range(args.hyperparam.discriminator_per_step):
            # Generate fake images, and train the model to predict them as fake. We keep track of the loss in predicting
            # fake images separately from real images.
            intv_dis_loss_fake += dis_model.train_on_batch(gen_model.predict(next(data.rand_vec)), data.label_fake)
            # Use real images, and train the model to predict them as real.
            intv_dis_loss_real += dis_model.train_on_batch(next(data.real), data.label_real)

        # Second, we train the generator for `step_gen` number of steps. Because the .trainable flag (for `dis_model`) 
        # was False when `com_model` was compiled, the discriminator weights are not updated. The generator weights are
        # the only weights updated in this step.
        # In this step, we train "generator" so that "discriminator(generator(random)) == real". Specifically, we compose
        # `dis_model` onto `gen_model`, and train the combined model so that given a random vector, it classifies images
        # as real.
        for _ in range(args.hyperparam.generator_per_step):
            intv_com_loss = np.add(intv_com_loss,
                                   com_model.train_on_batch(next(data.rand_vec), data.label_real))

        # That is the entire training algorithm.
        # Produce output every interval:
        if not batch % args.log_interval and batch != 0:
            logger.info("Completed batch {}/{}".format(batch, args.hyperparam.halt_batches))

            # Compute the average loss over this interval
            intv_com_loss      /= args.log_interval * args.hyperparam.generator_per_step
            intv_dis_loss_fake /= args.log_interval * args.hyperparam.discriminator_per_step
            intv_dis_loss_real /= args.log_interval * args.hyperparam.discriminator_per_step

            # Log a summary
            logger.info('Generator loss: {}.'.format(intv_com_loss))
            logger.info('Discriminator loss on real: {}, fake: {}.'.format(intv_dis_loss_real, intv_dis_loss_fake))

            # Log to CSV
            with open(config.get_filename('csv', args), 'a') as csvfile:
                print("{}, {}, {}, {}, {}".format(
                    datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                    batch,
                    intv_com_loss,
                    intv_dis_loss_real,
                    intv_dis_loss_fake), file=csvfile)
            
            # Zero out the running counters
            intv_com_loss[...]      = 0
            intv_dis_loss_fake[...] = 0
            intv_dis_loss_real[...] = 0

            # Write image
            img_fn = config.get_filename('image', args, batch)
            png.from_array(np.concatenate(data.unapply(gen_model.predict(next(data.rand_vec)))), 'RGB').save(img_fn)
            logger.debug("Saved sample images to {}.".format(img_fn))

            # Save weights
            gen_model.save_weights(config.get_filename('weight', args, 'gen', batch))
            dis_model.save_weights(config.get_filename('weight', args, 'dis', batch))
            logger.debug("Saved weights for batch {}.".format(batch))


if __name__ == '__main__':
    logger.info("Started")
    main(support.argparser().parse_args())
