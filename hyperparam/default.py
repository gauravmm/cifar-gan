# Hyperparameters for optimization

from ..typ import Hyperparameters

from keras import optimizers

param = Hyperparameters(
    optimizer    = optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999),  # as described in appendix A of DeepMind's AC-GAN paper
    halt_batches = 10000,
    batch_size   = 128,
    discriminator_per_step = 1,
    generator_per_step     = 4)