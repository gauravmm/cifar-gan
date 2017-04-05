# Hyperparameters for optimization

import numpy as np
from keras import optimizers

# As described in appendix A of DeepMind's AC-GAN paper
optimizer_gen = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
optimizer_dis = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
batch_size   = 128
discriminator_per_step = 1
generator_per_step     = 4

# This specifies the probability of a label being flipped, which allows the true and fake distributions to overlap
# to allow the GAN to discover this. There's an argument that (1) the space of good solutions is large and flat (i.e. 
# many solutions have similar likelihood.) (2) this makes it likely that the GAN finds a solution that isn't the best.
# We want to eventually implement an annealing schedule so we can control the rates of exploring and exploiting solutions.
# Here's a good resource: http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
label_flipping_prob = 0.05
# To disable this, just replace it with:
#   label_flipping_prob = 0.0

# This just makes labels noisy by adding gaussian noise.
label_smoothing  = lambda is_real, sz: np.random.normal(0,0.2,size=sz)
# To disable label smoothing noise, just replace it with:
#   lambda is_real, sz: 0
