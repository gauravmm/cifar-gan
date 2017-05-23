# Hyperparameters for optimization

import numpy as np
import tensorflow as tf
from support import MovingAverage

# Documentation is in the default hyperparmeter file.

SEED_DIM = (32,)
IMAGE_DIM = (32, 32, 3)
NUM_CLASSES = 10
BATCH_SIZE   = 64
LABELLED_FRACTION = 0.10
WGAN_ENABLE = True
WGAN_DIS_CLIP = 0.01

optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=0.00005)
optimizer_dis = tf.train.RMSPropOptimizer(learning_rate=0.00005)
optimizer_cls = tf.train.RMSPropOptimizer(learning_rate=0.00005)

# These noise and regularization methods are not required for WGANs:
label_flipping_prob = 0.0
label_smoothing = lambda is_real, sz: 0

loss_weights_generator = {'discriminator': 1.0, 'classifier': 0.0}
loss_weights_classifier = {'discriminator': 0.0, 'classifier': 1.0}

class HaltWGAN(object):
    def __init__(self):
        pass

    def discriminator_halt(self, batch, step, metrics):
        if batch < 25 and step < 100:
            return False
        elif batch % 200 == 0 and step < 100:
            return False
        else:
            return step >= 5

    def generator_halt(self, batch, step, metrics):
        return step >= 1

    def classifier_halt(self, batch, step, metrics):
        return step >= 1

_halting = HaltWGAN()
discriminator_halt  = _halting.discriminator_halt
generator_halt      = _halting.generator_halt
classifier_halt     = _halting.classifier_halt

ENABLE_TRAINING_DIS = True
ENABLE_TRAINING_CLS = False
ENABLE_TRAINING_GEN = True

SUMMARIZE_MORE = False