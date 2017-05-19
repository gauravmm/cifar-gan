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

optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=0.002)
optimizer_dis = tf.train.RMSPropOptimizer(learning_rate=0.002)
optimizer_cls = tf.train.RMSPropOptimizer(learning_rate=0.002)

label_flipping_prob = 0.0
label_smoothing  = lambda is_real, sz: 0

loss_weights_generator = {'discriminator': 1.0, 'classifier': 0.0}
loss_weights_classifier = {'discriminator': 0.0, 'classifier': 1.0}

discriminator_halt = lambda b, s, l: s >= 1
generator_halt     = lambda b, s, l: s >= 4
classifier_halt    = lambda b, s, l: s >= 1

ENABLE_TRAINING_DIS = True
ENABLE_TRAINING_CLS = False
ENABLE_TRAINING_GEN = False

SUMMARIZE_MORE = True