# Hyperparameters for optimization

import numpy as np
from keras import optimizers

# As described in appendix A of DeepMind's AC-GAN paper
optimizer    = optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999) 
halt_batches = 10000
batch_size   = 128
discriminator_per_step = 1
generator_per_step     = 4
label_noise  = lambda is_real, sz: np.random.normal(0,0.2,size=sz)
# To disable label noise, just replace it with:
#   lambda is_real, sz: 0
