# Hyperparameters for optimization

import numpy as np
from keras import optimizers

# Size of random seed used by the generator's input tensor:
SEED_DIM = (32,)
# Size of the output. The generator, discriminator and dataset will be required to use this size.
IMAGE_DIM = (32, 32, 3)
# Number of classes. The generator, discriminator and dataset will be required to use this size.
NUM_CLASSES = 10


# As described in appendix A of DeepMind's AC-GAN paper
optimizer_gen = optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=1e-8)
optimizer_dis = optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=1e-8)
#optimizers.SGD(lr=0.0002, decay=1e-8, momentum=0.9, nesterov=False)
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
        self.discriminator_correct = 0.60
        self.generator_correct = 0.60
        self.min_step_dis = 1
        self.max_step_dis = 3
        self.min_step_gen = 4
        self.max_step_gen = 12

    def discriminator_halt(self, batch, step, loss_fake, loss_real):
        # Batch refers to the number of times the discriminator, then generator would be training.
        # Step is the number of times the discriminator has been run within that batch
        # Loss metric the loss in the previous iteration, as a key:value dict.
        if step < self.min_step_dis:
            return False
        if step + 1 >= self.max_step_dis:
            return True
        if (loss_fake["label_fake"] + loss_real["label_real"])/2 < self.discriminator_correct:
            return True
        #if (loss_fake["loss"] + loss_real["loss"])/2 < self.discriminator_loss:
        #    return True
        return False

    def generator_halt(self, batch, step, loss):
        if step < self.min_step_gen:
            return False
        if step + 1 >= self.max_step_gen:
            return True
        if loss["label_real"] < self.generator_correct:
            return True
        #if loss["loss"] < self.generator_loss:
        #    return True
        return False

_halting = StepHalt()
discriminator_halt = lambda b, s, l1, l2: s >= 1 # _halting.discriminator_halt
generator_halt     = lambda b, s, l: s >= 4 # _halting.generator_halt
