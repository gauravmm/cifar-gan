# Hyperparameters for optimization

from keras import optimizers

# As described in appendix A of DeepMind's AC-GAN paper
optimizer    = optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999) 
halt_batches = 10000
batch_size   = 128
discriminator_per_step = 1
generator_per_step     = 4