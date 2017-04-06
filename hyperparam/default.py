# Hyperparameters for optimization

import numpy as np
from keras import optimizers

# As described in appendix A of DeepMind's AC-GAN paper
optimizer_gen = optimizers.Adam(lr=0.0005, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)
optimizer_dis = optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=False)
batch_size   = 128

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

class StepHalt(object):
    def __init__(self):
        self.discriminator_loss = 0.60
        self.generator_loss = 0.60
        self.min_step_dis = 1
        self.max_step_dis = 2
        self.min_step_gen = 3
        self.max_step_gen = 8

    def discriminator_halt(self, batch, step, loss_fake, loss_real):
        # Batch refers to the number of times the discriminator, then generator would be training.
        # Step is the number of times the discriminator has been run within that batch
        # Loss metric the loss in the previous iteration, as a key:value dict.
        if step < self.min_step_dis:
            return False
        if step >= self.max_step_dis:
            return True
        if (loss_fake["loss"] + loss_real["loss"])/2 < self.discriminator_loss:
            return True
        return False

    def generator_halt(self, batch, step, loss):
        if step < self.min_step_gen:
            return False
        if step >= self.max_step_gen:
            return True
        if loss["loss"] < self.generator_loss:
            return True
        return False

_halting = StepHalt()
discriminator_halt = _halting.discriminator_halt
generator_halt     = _halting.generator_halt
