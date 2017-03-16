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

import support
import tensorflow as tf
import numpy as np
from keras import layers, models, optimizers
from keras_diagram import ascii

#
# Init
#

np.random.seed(54183)

PATH = {
    "__main__": os.path.dirname(os.path.abspath(__file__)),
    "weights" : "weights",
    "logs"    : "train_logs",
    "cache"   : ".cache",
    "data"    : "data",
}
def WEIGHT_FILENAME(typ : str, name : str, step=None) -> str:
    if step is not None:
        step = "{:06d}".format(step) # Pad with leading zeros
    else:
        step = "*"                   # Wildcard
    
    return os.path.join(PATH["weights"], "checkpoint-{}-{}-{}".format(typ, name, step))

logging.basicConfig(filename=os.path.join(PATH['logs'], 'adversarial.log'), level=logging.DEBUG, format='[%(asctime)s, %(levelname)s] %(message)s')
# Logger
console = logging.StreamHandler()
console.setLevel(logging.INFO)
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
        # gen_model.load_weights(path_to_data, by_name=True)
        # dis_model.load_weights(path_to_data, by_name=True)
    else:
        # Delete old weight checkpoints
        for f in itertools.chain(glob.glob(WEIGHT_FILENAME("gen", args.generator.NAME)),
                                 glob.glob(WEIGHT_FILENAME("dis", args.discriminator.NAME))):
            logger.debug("Deleting weight file {}".format(f))
            os.remove(f)
        logger.info("Deleted all saved weights for generator \"{}\" and discriminator \"{}\".".format(args.generator.NAME, args.discriminator.NAME))
        batch = 0
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

    for i in range(nb_steps):
        logger.debug('Step: {} of {}.'.format(i, nb_steps))

        # This training proceeds in two phases: (1) discriminator, then (2) generator.
        # First, we train the discriminator for `step_dis` number of steps. Because the .trainable flag was True when 
        # `dis_model` was compiled, the weights of the discriminator will be updated. The discriminator is trained to
        # distinguish between "fake"" (generated) and real images by running it on one step of each.
        for _ in range(step_dis):
            # Generate fake images, and train the model to predict them as fake. We keep track of the loss in predicting
            # fake images separately from real images.
            batch_fake = gen_model.predict(rand_vec.next())
            intv_dis_loss_fake = np.add(intv_dis_loss_fake,
                                        dis_model.train_on_batch(generated_image_batch, label_fake))
            # Use real images, and train the model to predict them as real.
            batch_real = train_data.next()
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
                                   com_model.train_on_batch(rand_vec.next(), y_real))

        # That is the entire training algorithm.

        # Produce output every interval:
        if False: # not i % log_interval and i != 0:
            # plot batch of generated images w/ current generator
            figure_name = 'generated_image_batch_step_{}.png'.format(i)
            print('Saving batch of generated images at adversarial step: {}.'.format(i))

            generated_image_batch = generator_model.predict(np.random.normal(size=(batch_size, rand_dim)))
            real_image_batch = get_image_batch()

            plot_image_batch_w_labels.plot_batch(np.concatenate((generated_image_batch, real_image_batch)),
                                                 os.path.join(cache_dir, figure_name),
                                                 label_batch=['generated'] * batch_size + ['real'] * batch_size)

            # log loss summary
            print('Generator model loss: {}.'.format(combined_loss / (log_interval * step_gen)))
            print('Discriminator model loss real: {}.'.format(disc_loss_real / (log_interval * step_dis * 2)))
            print('Discriminator model loss generated: {}.'.format(disc_loss_generated / (log_interval * step_dis * 2)))

            combined_loss = np.zeros(shape=len(combined_model.metrics_names))
            disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
            disc_loss_generated = np.zeros(shape=len(discriminator_model.metrics_names))

            # save model checkpoints
            model_checkpoint_base_name = os.path.join(cache_dir, '{}_model_weights_step_{}.h5')
            generator_model.save_weights(model_checkpoint_base_name.format('generator', i))
            discriminator_model.save_weights(model_checkpoint_base_name.format('discriminator', i))


#
# Command-line handlers
#

# TODO: Typecheck Args
def dynLoadModule(pkg):
    # Used to dynamically load modules in commandline options.
    return lambda modname: importlib.import_module(pkg + "." + modname, package=".")

def argparser():
    parser = argparse.ArgumentParser(description='Train and test GAN models on data.')

    parser_g1 = parser.add_mutually_exclusive_group(required=True)
    parser_g1.add_argument('--train', action='store_const', dest='split', const='train', default='')
    parser_g1.add_argument('--test', action='store_const', dest='split', const='test', default='')

    parser.add_argument('--data', metavar='D', default="cifar10",
        type=dynLoadModule("data"),
        help='the name of a tf.slim dataset reader in the data package')
    parser.add_argument('--preprocessing', metavar='D', default="default",
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

    return parser

if __name__ == '__main__':
    logger.info("Started")

    
    args = argparser().parse_args()

    main(args)
