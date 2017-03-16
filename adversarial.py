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

from keras import layers, models, optimizers
from keras_diagram import ascii

#
# Init
#

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
k_d = 1  # number of discriminator network updates per step
k_g = 2  # number of generative network updates per step
log_interval = 100  # interval (in steps) at which to log loss summaries & save plots of image samples to disc


def main(args):
    print(args)
    logger.info("Loaded dataset      : \t{}".format(args.data.__file__))
    logger.info("Loaded generator    : \t{}".format(args.generator.__file__))
    logger.info("Loaded discriminator: \t{}".format(args.discriminator.__file__))

    img_dim = args.generator.IMAGE_DIM


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
    dis_model.trainable = False # TODO: I don't understand why this is set.
    com_model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])

    logger.info("Compiled models.")
    logger.debug("Generative model structure:\n{}".format(ascii(gen_model)))
    logger.debug("Discriminative model structure:\n{}".format(ascii(dis_model)))

    # Load weights if necessary
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
            logger.debug("Delete weight file {}".format(f))
            os.remove(f)
        logger.info("Deleted all saved weights for generator \"{}\" and discriminator \"{}\".".format(args.generator.NAME, args.discriminator.NAME))




def adversarial_training(data_dir, generator_model_path, discriminator_model_path):
    #
    # data generators
    #

    data_generator = image.ImageDataGenerator(
        preprocessing_function=applications.xception.preprocess_input,
        dim_ordering='tf')

    flow_from_directory_params = {'target_size': (img_height, img_width),
                                  'color_mode': 'grayscale' if img_channels == 1 else 'rgb',
                                  'class_mode': None,
                                  'batch_size': batch_size}

    real_image_generator = data_generator.flow_from_directory(
        directory=data_dir,
        **flow_from_directory_params
    )

    def get_image_batch():
        img_batch = real_image_generator.next()

        # keras generators may generate an incomplete batch for the last batch
        if len(img_batch) != batch_size:
            img_batch = real_image_generator.next()

        assert img_batch.shape == (batch_size, img_height, img_width, img_channels), img_batch.shape
        return img_batch

    # the target labels for the binary cross-entropy loss layer are 0 for every yj (real) and 1 for every xi (generated)
    y_real = np.array([0] * batch_size)
    y_generated = np.array([1] * batch_size)

    combined_loss = np.zeros(shape=len(combined_model.metrics_names))
    disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
    disc_loss_generated = np.zeros(shape=len(discriminator_model.metrics_names))


    for i in range(nb_steps):
        print('Step: {} of {}.'.format(i, nb_steps))

        # train the discriminator
        for _ in range(k_d):
            generator_input = np.random.normal(size=(batch_size, rand_dim))
            # sample a mini-batch of real images
            real_image_batch = get_image_batch()

            # generate a batch of images with the current generator
            generated_image_batch = generator_model.predict(generator_input)

            # update φ by taking an SGD step on mini-batch loss LD(φ)
            disc_loss_real = np.add(discriminator_model.train_on_batch(real_image_batch, y_real), disc_loss_real)
            disc_loss_generated = np.add(discriminator_model.train_on_batch(generated_image_batch, y_generated),
                                         disc_loss_generated)

        # train the generator
        for _ in range(k_g * 2):
            generator_input = np.random.normal(size=(batch_size, rand_dim))

            # update θ by taking an SGD step on mini-batch loss LR(θ)
            combined_loss = np.add(combined_model.train_on_batch(generator_input, y_real), combined_loss)

        if not i % log_interval and i != 0:
            # plot batch of generated images w/ current generator
            figure_name = 'generated_image_batch_step_{}.png'.format(i)
            print('Saving batch of generated images at adversarial step: {}.'.format(i))

            generated_image_batch = generator_model.predict(np.random.normal(size=(batch_size, rand_dim)))
            real_image_batch = get_image_batch()

            plot_image_batch_w_labels.plot_batch(np.concatenate((generated_image_batch, real_image_batch)),
                                                 os.path.join(cache_dir, figure_name),
                                                 label_batch=['generated'] * batch_size + ['real'] * batch_size)

            # log loss summary
            print('Generator model loss: {}.'.format(combined_loss / (log_interval * k_g * 2)))
            print('Discriminator model loss real: {}.'.format(disc_loss_real / (log_interval * k_d * 2)))
            print('Discriminator model loss generated: {}.'.format(disc_loss_generated / (log_interval * k_d * 2)))

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
