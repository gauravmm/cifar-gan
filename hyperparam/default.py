# Hyperparameters for optimization

import numpy as np
import tensorflow as tf
from support import MovingAverage

# Size of random seed used by the generator's input tensor:
SEED_DIM = (32,)
# Size of the output. The generator, discriminator and dataset will be required to use this size:
IMAGE_DIM = (32, 32, 3)
# Number of classes. The generator, discriminator and dataset will be required to use this size:
NUM_CLASSES = 10
# Number of images per batch:
BATCH_SIZE   = 64
# Semi-supervised data fraction:
LABELLED_FRACTION = 0.10
# WGAN Compatibility:
# If you set this flag, you also need to:
#   1) set WGAN_DIS_CLIP to a small value (~0.01), and--
#   2) set the *_halt functions appropriately.
WGAN_ENABLE = False
# In the training step of a WGAN, the discriminator weights will be clipped to (-WGAN_DIS_CLIP, WGAN_DIS_CLIP).
# If WGAN_ENABLE is False, this parameter is not required.
WGAN_DIS_CLIP = 0.01
# To enable an additional l2-loss on the magnitude of all trainable weights, set this to a non-zero number. This is added
# to all losses. This will not work with WGANs.
WEIGHT_DECAY - 0.01

optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=0.002)
optimizer_dis = tf.train.RMSPropOptimizer(learning_rate=0.002)
optimizer_cls = tf.train.RMSPropOptimizer(learning_rate=0.002)

# This specifies the probability of a label being flipped, which allows the true and fake distributions to overlap
# to allow the GAN to discover this. There's an argument that (1) the space of good solutions is large and flat (i.e. 
# many solutions have similar likelihood.) (2) this makes it likely that the GAN finds a solution that isn't the best.
# We want to eventually implement an annealing schedule so we can control the rates of exploring and exploiting solutions.
# Here's a good resource: http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
label_flipping_prob = 0.1
# To disable this, just replace it with:
#   label_flipping_prob = 0.0

# This just makes labels noisy by adding gaussian noise.
label_smoothing  = lambda is_real, sz: np.random.normal(0,0.2,size=sz)
# To disable label smoothing noise, just replace it with:
#   lambda is_real, sz: 0

# The relative weight assigned to the discriminator and classifier nodes when training the generator network
# and the classifier network respectively.
loss_weights_generator = {'discriminator': 1.0, 'classifier': 1.0}
loss_weights_classifier = {'discriminator': 0.0, 'classifier': 1.0}

class HaltRelativeCorrectness(object):
    def __init__(self):
        self.discriminator_correct = 0.51
        self.generator_correct = 0.51
        self.classifier_correct = 0.3
        self.min_step_dis = 1
        self.max_step_dis = 6
        self.min_step_gen = 4
        self.max_step_gen = 12
        self.min_step_cls = 1
        self.max_step_cls = 3

    def discriminator_halt(self, batch, step, metrics):
        # Batch refers to the number of times the discriminator, then generator would be training.
        # Step is the number of times the discriminator has been run within that batch
        # Metric the loss statistics in the previous iteration, as a key:value dict.
        if step < self.min_step_dis:
            return False
        if step + 1 >= self.max_step_dis:
            return True
        if metrics["real_true_pos"] < self.discriminator_correct:
            return False
        if metrics["fake_true_neg"] < self.discriminator_correct:
            return False
        return True

    def generator_halt(self, batch, step, metrics):
        if step < self.min_step_gen:
            return False
        if step + 1 >= self.max_step_gen:
            return True
        if metrics["gen_fooling"] > self.generator_correct:
            return True
        return False

    # classifier_halt is also called when step=0, because the classifier may not have to be run every batch.
    def classifier_halt(self, batch, step, metrics):
        if step < self.min_step_cls:
            return False
        if step + 1 >= self.max_step_cls:
            return True
        if metrics["cls_accuracy"] > self.classifier_correct:
            return True
        return False


_halting = HaltRelativeCorrectness()

discriminator_halt  = _halting.discriminator_halt
generator_halt      = _halting.generator_halt
classifier_halt     = _halting.classifier_halt

# Naive halting criteria:
#discriminator_halt = lambda b, s, l: s >= 1
#generator_halt     = lambda b, s, l: s >= 6
#classifier_halt    = lambda b, s, l: s >= 1

ENABLE_TRAINING_DIS = True
ENABLE_TRAINING_CLS = True
ENABLE_TRAINING_GEN = True

# If this is true, add more items to the training summaries.
SUMMARIZE_MORE = False

# Unused, but a good example:
"""
class HaltRelativeLoss(object):
    def __init__(self):
        self.gen_to_dis_ratio = lambda gen_loss, dis_loss: 4.5 + 2.*(gen_loss-dis_loss)/dis_loss
        self.gen_loss = MovingAverage(30)
        self.dis_loss = MovingAverage(30)
        self.last_batch = -1
        self.curr_gen_ratio = 0
        self.dis_steps = 2  # STOP AT TWO!
        self.cls_steps = 3

    def discriminator_halt(self, batch, step, metrics):
        # Batch refers to the number of times the discriminator, then generator would be training.
        # Step is the number of times the discriminator has been run within that batch
        # Loss metric the loss in the previous iteration, as a key:value dict.
        self.dis_loss.push(0.5*(metrics["fake_loss"] + metrics["real_loss"]))
        return step >= self.dis_steps

    def generator_halt(self, batch, step, metrics):
        if batch > self.last_batch:
            self.last_batch = batch
            self.curr_gen_ratio = self.gen_to_dis_ratio(self.gen_loss.get(), self.dis_loss.get())

        self.gen_loss.push(loss["loss"])
        return step >= self.curr_gen_ratio * self.dis_steps
    
    def classifier_halt(self, batch, step, metrics):
        return step >= self.cls_steps
"""
