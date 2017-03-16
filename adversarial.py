#!/bin/python3

"""
Main CIFAR-GAN file.
Handles the loading of data from ./data, models from ./models, training, and testing.

Modified from TensorFlow-Slim examples and https://github.com/wayaai/GAN-Sandbox/
"""

import argparse, importlib, sys, os

#
# Init
#

PATH = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PATH, '.cache')


def main(args):
    print(args)


def adversarial_training(data_dir, generator_model_path, discriminator_model_path):
    """
    Adversarial training of the generator network Gθ and discriminator network Dφ.

    """
    #
    # define model input and output tensors
    #

    generator_input_tensor = layers.Input(shape=(rand_dim,))
    generated_image_tensor = generator_network(generator_input_tensor)

    generated_or_real_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    discriminator_output = discriminator_network(generated_or_real_image_tensor)

    #
    # define models
    #

    generator_model = models.Model(input=generator_input_tensor, output=generated_image_tensor, name='generator')
    discriminator_model = models.Model(input=generated_or_real_image_tensor, output=discriminator_output,
                                       name='discriminator')

    combined_output = discriminator_model(generator_model(generator_input_tensor))
    combined_model = models.Model(input=generator_input_tensor, output=combined_output, name='combined')

    #
    # compile models
    #

    adam = optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)  # as described in appendix A of DeepMind's AC-GAN paper

    generator_model.compile(optimizer=adam, loss='binary_crossentropy')
    discriminator_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    discriminator_model.trainable = False
    combined_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    print(generator_model.summary())
    print(discriminator_model.summary())

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

    if generator_model_path:
        generator_model.load_weights(generator_model_path, by_name=True)
    if discriminator_model_path:
        discriminator_model.load_weights(discriminator_model_path, by_name=True)

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

def dynLoadModule(pkg):
    # Used to dynamically load modules in commandline options.
    return lambda modname: importlib.import_module(pkg + "." + modname, package=".")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test GAN models on data.')

    parser_g1 = parser.add_mutually_exclusive_group(required=True)
    parser_g1.add_argument('--train', action='store_const', dest='split', const='train', default='')
    parser_g1.add_argument('--test', action='store_const', dest='split', const='test', default='')

    parser.add_argument('--data', metavar='D', default="cifar10", type=dynLoadModule("data"),
                    help='the name of a tf.slim dataset reader in the data package')
    parser.add_argument('--preprocessing', metavar='D', default="default", type=dynLoadModule("preprocessing"),
                    help='the name of a tf.slim dataset reader in the data package')
    parser.add_argument('--generator', metavar='G', type=dynLoadModule("models"),
                    help='name of the module containing the generator model definition')
    parser.add_argument('--discriminator', metavar='S', type=dynLoadModule("models"),
                    help='name of the module containing the discrimintator model definition')
    args = parser.parse_args()

    main(args)
