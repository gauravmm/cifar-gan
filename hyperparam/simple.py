# Hyperparameters for optimization

import numpy as np
import tensorflow as tf
from support import MovingAverage

SEED_DIM = (32,)
IMAGE_DIM = (32, 32, 3)
NUM_CLASSES = 10
BATCH_SIZE   = 64
LABELLED_FRACTION = 0.10
WGAN_ENABLE = False
# To enable an additional l2-loss on the magnitude of all trainable weights, set this to a non-zero number.
WEIGHT_DECAY = 0.0

optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=0.002)
optimizer_dis = tf.train.RMSPropOptimizer(learning_rate=0.002)
optimizer_cls = tf.train.RMSPropOptimizer(learning_rate=0.002)

label_flipping_prob = 0.1
label_smoothing  = lambda is_real, sz: np.random.normal(0,0.1,size=sz)

loss_weights_generator = {'discriminator': 1.0, 'classifier': 0.0}
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
        if metrics["gen_fooling"] < self.generator_correct:
            return False
        return True

    def classifier_halt(self, batch, step, metrics):
        if step < self.min_step_cls:
            return False
        if step + 1 >= self.max_step_cls:
            return True
        if metrics["cls_accuracy"] < self.classifier_correct:
            return False
        return True


_halting = HaltRelativeCorrectness()

discriminator_halt  = _halting.discriminator_halt
generator_halt      = _halting.generator_halt
classifier_halt     = _halting.classifier_halt

ENABLE_TRAINING_DIS = True
ENABLE_TRAINING_CLS = True
ENABLE_TRAINING_GEN = True

# If this is true, add more items to the training summaries.
SUMMARIZE_MORE = False
