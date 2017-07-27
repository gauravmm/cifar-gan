# Hyperparameters for optimization
# Contains all you need to change for the experiments

import numpy as np
import tensorflow as tf
from support import MovingAverage

SEED_DIM = (100,) #dimension of the input noise vector of the Generator
IMAGE_DIM = (32, 32, 3)
NUM_CLASSES = 10
BATCH_SIZE   = 100
LABELLED_FRACTION = 0.1 #percentage of labels of the training dataset that we actually use
WGAN_ENABLE = False #do you want to use Wasserstrein GANs ?
WEIGHT_DECAY = 0.0001
LR_GEN = 0.0003 #learning rate of the Generator
LR_DIS = 0.0003 #learning rate of the Discriminator
LR_CLS = 0.0003 #learning rate of the Classifier
MOM1 = 0.5 #first momentum parameter of the Adam optimizer

#In case you are not using the OpenAI-based GAN, the gradient descent optimizers are defined here:
#(otherwise, they are set to Adam)
optimizer_gen = tf.train.AdamOptimizer(learning_rate=LR_GEN)
optimizer_dis = tf.train.AdamOptimizer(learning_rate=LR_DIS)
optimizer_cls = tf.train.AdamOptimizer(learning_rate=LR_CLS)

label_flipping_prob = 0.0 #percentage of the generated/real data which labels are flipped
label_smoothing  = lambda is_real, sz: np.random.normal(0,0.0,size=sz) #smoothening the 0,1 labels

loss_weights_generator = {'discriminator': 1.0, 'classifier': 1.0} #the Generator is updated from both the Discriminator and the Classifier, with coefficients that you set
loss_weights_classifier = {'discriminator': 0.0, 'classifier': 1.0} #same for the Classifier 

#here we define the training schedule at each iteration, one iteration being one batch
#There are 2 ways of doing that:
#1: set an objective for each network to reach 
#2: set a minimum and maximum number of steps (if thye 2 are equal, then the objectives defined earlier do no make sense anymore)
class HaltRelativeCorrectness(object):
    def __init__(self):
        #objectives
        self.discriminator_correct = 0.53 #percentage of the time that D needs to fool G before we stop training it
        self.generator_correct = 0.53 #percentage of the time that G needs to fool D before we stop training it
        self.classifier_min_correct = 0.8 #minimum accuracy that C needs to reach
        self.classifier_max_correct = 0.98 #maximum accuracy that C must not get above of (#overfitting)
        self.min_step_dis = 1 #minimum number of steps to train D for at each iteration
        self.max_step_dis = 1 #maximum number of steps to train D for at each iteration
        self.min_step_gen = 1 #minimum number of steps to train G for at each iteration 
        self.max_step_gen = 1 #maximum number of steps to train G for at each iteration
        self.min_step_cls = 1 #minimum number of steps to train C for at each iteration
        self.max_step_cls = 1 #maximum number of steps to train C for at each iteration

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
        if metrics["cls_accuracy"] < self.classifier_min_correct:
            return False
        if metrics["cls_accuracy"] > self.classifier_max_correct:
            return True
        return True


_halting = HaltRelativeCorrectness()

discriminator_halt  = _halting.discriminator_halt
generator_halt      = _halting.generator_halt
classifier_halt     = _halting.classifier_halt

#Do you want to train D ?
ENABLE_TRAINING_DIS = True
#Do you want to train C ?
ENABLE_TRAINING_CLS = True
#do you want to train G ? 
ENABLE_TRAINING_GEN = True

# If this is true, add more items to the training summaries.
SUMMARIZE_MORE = False
