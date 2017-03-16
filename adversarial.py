#!/bin/python3

"""
Main CIFAR-GAN file.
Handles the loading of data from ./data, models from ./models, training, and testing.

Modified from TensorFlow-Slim examples and https://github.com/wayaai/GAN-Sandbox/
"""

import argparse
import glob
import importlib
import itertools
import logging
import os
import sys
import config

import numpy as np

import png
import support
import tensorflow as tf
from keras import layers, models, optimizers
from keras_diagram import ascii

#
# Init
#

np.random.seed(54183)

# Logging
logging.basicConfig(filename=os.path.join(config.PATH['logs'], 'adversarial.log'), level=logging.DEBUG, format='[%(asctime)s, %(levelname)s] %(message)s')
# Logger
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s] %(message)s', datefmt='%H:%M:%S'))
logging.getLogger().addHandler(console)
logger = logging.getLogger()


# TODO: Abstract the optimizer and training parameters out, possibly into the model definition, possibly elsewhere
optim = optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)  # as described in appendix A of DeepMind's AC-GAN paper
# training params
nb_steps = 10000
batch_size = 128
step_dis = 1  # number of discriminator network updates per step
step_gen = 4  # number of generative network updates per step
log_interval = 100  # interval (in steps) at which to log loss summaries & save plots of image samples to disc


def main(args):
    print(args)
    logger.info("Loaded dataset      : \t{}".format(args.data.__file__))
    logger.info("Loaded generator    : \t{}".format(args.generator.__file__))
    logger.info("Loaded discriminator: \t{}".format(args.discriminator.__file__))

    img_dim = args.generator.IMAGE_DIM


    #
    # Build Model
    #

    # TODO: Update calls to Keras 2 API to remove warnings. We may have to restructure the code accordingly
    # See use of new API here: https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

    # Input/output tensors:
    gen_input  = layers.Input(shape=args.generator.SEED_DIM)
    gen_output = args.generator.generator(gen_input, img_dim)
    dis_input  = layers.Input(shape=args.generator.IMAGE_DIM)
    dis_output = args.discriminator.discriminator(dis_input, img_dim)

    logger.info("Constructed computational graphs.")


    # Define and compile models
    gen_model = models.Model(input=gen_input, output=gen_output, name='generator')
    dis_model = models.Model(input=dis_input, output=dis_output, name='discriminator')
    # We compose the discriminator onto the generator to produce the combined model:
    com_model = models.Model(input=gen_input, output=dis_model(gen_model(gen_input)), name='combined')

    gen_model.compile(optimizer=optim, loss='binary_crossentropy')
    dis_model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
    # The trainable flag only takes effect upon compilation. By setting it False here, we allow the discriminator weights
    # to be updated in the step where we learn dis_model directly (compiled above), but not in the step where we learn
    # gen_model (compiled below). This behaviour is important, see comments in the training loop for details.
    dis_model.trainable = False 
    com_model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])

    logger.info("Compiled models.")
    logger.debug("Generative model structure:\n{}".format(ascii(gen_model)))
    logger.debug("Discriminative model structure:\n{}".format(ascii(dis_model)))


    #
    # Load weights
    #

    batch = None
    if args.resume:
        assert not "This feature is not implemented yet!"
        # TODO: Implement resumed training, loading the file with the highest step number that matches the wildcard
        # from WEIGHT_FILENAME().
        support.get_latest_blob(config.get_filename('weight', args, 'gen'))
        # gen_model.load_weights(path_to_data, by_name=True)
        # dis_model.load_weights(path_to_data, by_name=True)
    else:
        # Delete old weight checkpoints
        for f in itertools.chain(glob.glob(config.get_filename('weight', args)),
                                 glob.glob(config.get_filename('image', args))):
            logger.debug("Deleting file {}".format(f))
            os.remove(f)
        logger.info("Deleted all saved weights and images for generator \"{}\" and discriminator \"{}\".".format(args.generator.NAME, args.discriminator.NAME))
        batch = 0

        with open(config.get_filename('csv', args), 'wb') as csvfile:
            csvfile.write("time, batch, composite loss, discriminator+real loss, discriminator+fake loss\n")

    assert batch is not None
    

    #
    # Load data
    #
    
    # TODO: Wrap the data source and transformations properly
    # NOTE: Keras requires Numpy input, so we cannot use Tensorflow's built-in data augmentation tools. We can instead use Keras' tools.
    train_data, train_labels = args.data.get_data("train")
    train_data = support.data_stream(train_data, batch_size)
    # We don't need the labels for this:
    # train_labels = support.data_stream(train_labels, batch_size)
    # Random vector to feed generator:
    rand_vec = support.random_stream(batch_size, args.generator.SEED_DIM)
    logger.debug("Data loaded from disk.")


    #
    # Training
    #

    # Create virual data
    label_real = np.array([0] * batch_size) # Label to train discriminator on real data
    label_fake = np.array([1] * batch_size) # Label to train discriminator on generated data

    # Loss value in the current log interval:
    intv_com_loss = np.zeros(shape=len(com_model.metrics_names))
    intv_dis_loss_real = np.zeros(shape=len(dis_model.metrics_names))
    intv_dis_loss_fake = np.copy(intv_dis_loss_real)

    for batch in range(batch, nb_steps):
        logger.info('Step {} of {}.'.format(batch, nb_steps))

        # This training proceeds in two phases: (1) discriminator, then (2) generator.
        # First, we train the discriminator for `step_dis` number of steps. Because the .trainable flag was True when 
        # `dis_model` was compiled, the weights of the discriminator will be updated. The discriminator is trained to
        # distinguish between "fake"" (generated) and real images by running it on one step of each.
        for _ in range(step_dis):
            # Generate fake images, and train the model to predict them as fake. We keep track of the loss in predicting
            # fake images separately from real images.
            batch_fake = gen_model.predict(next(rand_vec))
            intv_dis_loss_fake = np.add(intv_dis_loss_fake,
                                        dis_model.train_on_batch(generated_image_batch, label_fake))
            # Use real images, and train the model to predict them as real.
            batch_real = next(train_data)
            intv_dis_loss_real = np.add(intv_dis_loss_real,
                                        dis_model.train_on_batch(real_image_batch, label_real))

        # Second, we train the generator for `step_gen` number of steps. Because the .trainable flag (for `dis_model`) 
        # was False when `com_model` was compiled, the discriminator weights are not updated. The generator weights are
        # the only weights updated in this step.
        # In this step, we train "generator" so that "discriminator(generator(random)) == real". Specifically, we compose
        # `dis_model` onto `gen_model`, and train the combined model so that given a random vector, it classifies images
        # as real.
        for _ in range(step_gen):
            intv_com_loss = np.add(intv_com_loss,
                                   com_model.train_on_batch(next(rand_vec), y_real))

        # That is the entire training algorithm.

        # Produce output every interval:
        if not batch % log_interval and i != 0:
            logger.debug("Logging at batch {}/{}".format(batch, nb_steps))

            # Compute and log loss
            intv_com_loss /= log_interval * step_gen
            intv_dis_loss_fake /= log_interval * step_dis
            intv_dis_loss_real /= log_interval * step_dis

            # log loss summary
            logger.debug('Generator loss: {}.'.format(intv_com_loss))
            logger.debug('Discriminator loss on real: {}, fake: {}.'.format(intv_dis_loss_real, intv_dis_loss_fake)

            # Log to CSV
            with open(config.get_filename('csv', args), 'a') as csvfile:
                csvfile.write("{}s, {}d, {}f, {}f, {}f\n".format(
                    datetime.isoformat(),
                    batch,
                    intv_com_loss,
                    intv_dis_loss_real,
                    intv_dis_loss_fake))
            
            intv_com_loss = 0
            intv_dis_loss_fake = 0
            intv_dis_loss_real = 0

            # Write image
            img_fn = config.get_filename('image', args, batch)
            png.from_array(np.concatenate(gen_model.predict(next(rand_vec))), 'RGB').save(img_fn)
            logger.debug("Saved sample images to {}.".format(img_fn))

            # Save weights
            gen_model.save_weights(config.get_filename('weight', args, 'gen', batch))
            dis_model.save_weights(config.get_filename('weight', args, 'dis', batch))

#
# Command-line handlers
#

# TODO: Typecheck Args
def dynLoadModule(pkg):
    # Used to dynamically load modules in commandline options.
    return lambda modname: importlib.import_module(pkg + "." + modname, package=".")

def argparser():
    parser = argparse.ArgumentParser(description='Train and test GAN models on data.')

    parser.add_argument('--data', metavar='D', default="cifar10",
        type=dynLoadModule("data"),
        help='the name of a tf.slim dataset reader in the data package')
    parser.add_argument('--preprocessing', nargs="*",
        type=dynLoadModule("preprocessing"),
        help='the name of a tf.slim dataset reader in the data package')
    parser.add_argument('--generator', metavar='G', 
        type=dynLoadModule("models"),
        help='name of the module containing the generator model definition')
    parser.add_argument('--discriminator', metavar='S',
        type=dynLoadModule("models"),
        help='name of the module containing the discrimintator model definition')
    parser.add_argument('--resume', action='store_const', const=True, default=False,
        help='attempt to load saved weights and continue training')
    parser.add_argument("split", choices=["train", "test"])

    return parser

if __name__ == '__main__':
    logger.info("Started")

    
    args = argparser().parse_args()

    main(args)
