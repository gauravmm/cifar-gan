# Data support functions

import argparse
import functools
import glob
import importlib
import itertools
import logging
import os
import sys

import numpy as np

import config
from config import IMAGE_GUTTER
from typing import Tuple

logger = logging.getLogger()

#
# Data
#

class Data(object):
    def __init__(self, args):
        train_data, train_labels = args.data.get_data("train")
        logger.info("Training data loaded from disk.")

        train = zip(_data_stream(train_data, args.hyperparam.batch_size), _data_stream(train_labels, args.hyperparam.batch_size))

        # We apply all the preprocessors in order to get a generator that automatically applies preprocessing.
        for p in args.preprocessor:
            train = itertools.starmap(p.apply, train)
        # Only keep the images, discard the labels
        train = map(lambda x: x[0], train)
        
        self.rand_vec = _random_stream(args.hyperparam.batch_size, args.generator.SEED_DIM)
        # Present images them in chunks of exactly batch-size:
        self.real     = _image_stream_batch(train, args.hyperparam.batch_size)
        self.raw      = (train_data, train_labels)
        self.unapply  = functools.reduce(lambda f, g: lambda x: f(g(x)), [p.unapply for p in reversed(args.preprocessor)], lambda x: x)

        # Use to label a discriminator batch as real
        self._label_dis_real = map(lambda a, b: a + b,
                              _value_stream(args.hyperparam.batch_size, 0),
                              _function_stream(lambda: args.hyperparam.label_smoothing(True, args.hyperparam.batch_size)))
        # Use to label a discriminator batch as fake
        self._label_dis_fake = map(lambda a, b: a + b,
                              _value_stream(args.hyperparam.batch_size, 1),
                              _function_stream(lambda: args.hyperparam.label_smoothing(False, args.hyperparam.batch_size)))
        # Random flipping support
        self.label_dis_real = _selection_stream([args.hyperparam.label_flipping_prob],
                                                self._label_dis_real, self._label_dis_fake)
        self.label_dis_fake = _selection_stream([args.hyperparam.label_flipping_prob],
                                                self._label_dis_fake, self._label_dis_real)
        # Use to label a generator batch as real
        self.label_gen_real = _value_stream(args.hyperparam.batch_size, 1)

# TODO: Support reading test data.
# TODO: Change convention so that the data class returns a generator. We can work with generators all the way down for
#       better generalization.
# TODO: Support randomization of input

# A generator that enforces the batch-size of the input. Used to feed keras the right amount of data even with data 
# augmentation increasing the batch size.
def _image_stream_batch(itr, batch_size):
    remainder = next(itr)
    while True:
        while remainder.shape[0] < batch_size:
            remainder = np.concatenate((remainder, next(itr)))
        yield remainder[:batch_size,...]
        remainder = remainder[batch_size:,...]

def _data_stream(dataset, batch_size : int):
    # The first index of the next batch:
    i = 0 # Type: int
    
    while True:
        j = i + batch_size
        # If we wrap around the back of the dataset:
        if j >= dataset.shape[0]:
            x = list(range(i, dataset.shape[0])) + list(range(0, j - dataset.shape[0]))
            yield dataset[x,...]
            i = j - dataset.shape[0]
        else:
            yield dataset[i:j,...]
            i = j
    return data_gen

# Produces a stream of random data
def _random_stream(batch_size : int, img_size : Tuple[int, int, int]):
    sz = [batch_size, img_size[0]]
    while True:
        yield np.random.normal(size=sz)

def _value_stream(batch_size : int, value):
    while True:
        yield np.full((batch_size,), value, dtype=np.float16)

def _log_stream(name, gen):
    while True:
        g = next(gen)
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        logger.debug("Logging values in stream {}: {}".format(name, g))
        yield g

def _function_stream(func):
    while True:
        yield func()

def _selection_stream(probs, *args):
    """
    Implements elementwise random.choice; For each element in the generators passed to args, selects them according to
    the distribution given in probs. If none is given, uniform distribution is used.
    """
    if len(args) == 0:
        raise "We need at least one arg to select from."

    if probs is not None:
        probs = list(probs)
    else:
        probs = [1/float(len(args))] * args
    s = sum(probs)

    if s > 1:
        raise AssertionError("The sum of all probabilities given to selection stream must be no more than 1.")
    n = len(args) - len(probs)
    for _ in range(n):
        probs.append((1-s)/float(n))

    while True:
        # Get the next element for all the generators in args. We assemble this into a list of ndarray, where each ndarray
        # is of shape (i, j, k, ...).
        ch = list(map(lambda g: next(g), args))
        
        # The call to np.random.choice makes ndarray of shape (i, j, k, ...), each element containing a value from
        # range(len(args)), selected from the discrete distribution given by probs. We then use np.choose to map that
        # into an actual value.
        yield np.choose(np.random.choice(len(ch), size=ch[0].shape, p=probs), ch)


#
# Image handling
#

def arrange_images(img_data, args):
    num, rw, cl, ch = img_data.shape
    cols = args.image_columns
    rows = num // args.image_columns

    rv = np.empty((rows * (rw + IMAGE_GUTTER) - IMAGE_GUTTER, cols * (cl + IMAGE_GUTTER) - IMAGE_GUTTER, ch), dtype=np.uint8)
    rv[...] = 255

    for i in range(rows):
        for j in range(cols):
            rv[i*(rw + IMAGE_GUTTER):i*(rw + IMAGE_GUTTER) + rw,
               j*(cl + IMAGE_GUTTER):j*(cl + IMAGE_GUTTER) + cl,:] = img_data[i * cols + j,...]
    
    return rv

#
# Filesystem
#

def _get_latest_glob(blob):
    """
    Returns the file that matches blob (with a single wildcard), that has the highest numeric value in the wildcard.
    """
    assert len(list(filter(lambda x: x == "*", blob))) == 1 # There should only be one wildcard in blob
    
    blobs = glob.glob(blob)
    if not len(blobs):
        raise Exception("Cannot file file matching {}".format(blob))
    
    ltrunc = blob.index("*")           # Number of left characters to remove
    rtrunc = -(len(blob) - ltrunc - 1) # Number of right characters to remove
    
    # Get the indices hidden behind the wildcard
    idx = [int(b[ltrunc:rtrunc]) for b in blobs]
    return sorted(zip(idx, blobs), reverse=True)[0]

def resume(args, gen_model, dis_model):
    try:
        # Load files as necessary
        gen_num, gen_fn = _get_latest_glob(config.get_filename('weight', args, 'gen'))
        dis_num, dis_fn = _get_latest_glob(config.get_filename('weight', args, 'dis'))

        # Check if the files are from the same batch.
        assert gen_num == dis_num

        gen_model.load_weights(gen_fn, by_name=True)
        logger.info("Loaded generator weights from {}".format(gen_fn))
        dis_model.load_weights(dis_fn, by_name=True)
        logger.info("Loaded discriminator weights from {}".format(dis_fn))

        return gen_num + 1

    except Exception as e:
        logger.warn("Exception: {}".format(e))
        logger.debug(sys.exc_info())
        return None

def clear(args):
    # Delete old weight checkpoints
    for f in itertools.chain(glob.glob(config.get_filename('weight', args)),
                             glob.glob(config.get_filename('image',  args))):
        logger.debug("Deleting file {}".format(f))
        os.remove(f)
    logger.info("Deleted all saved weights and images for generator \"{}\" and discriminator \"{}\".".format(args.generator.NAME, args.discriminator.NAME))

    return 0

#
# CLI
#

def dynLoadModule(pkg):
    # Used to dynamically load modules in commandline options.
    return lambda modname: importlib.import_module(pkg + "." + modname, package="")

def argparser():
    parser = argparse.ArgumentParser(description='Train and test GAN models on data.')

    parser.add_argument('--preprocessor', nargs="*", default=[],
        type=dynLoadModule("preprocessor"),
        help='the name of files with image preprocessing instructions in the preprocessor package; applied in left-to-right order')
    parser.add_argument('--log-interval', default=config.LOG_INTERVAL_DEFAULT, type=int,
        help="the number of batches between reporting results and saving weights")
    parser.add_argument('--image-columns', default=1, type=int,
        help="the number of columns to group produced images into")        
    parser.add_argument('--resume', action='store_const', const=True, default=False,
        help='attempt to load saved weights and continue training')

    parser.add_argument('--data', default="cifar10",
        type=dynLoadModule("data"),
        help='the name of a file in the data package, used to specify the dataset loader')
    parser.add_argument('--hyperparam', default="default",
        type=dynLoadModule("hyperparam"),
        help='the name of a hyperparameter definition file in the hyperparam package')
    parser.add_argument('--generator', required=True, 
        type=dynLoadModule("models"),
        help='name of the file containing the generator model definition')
    parser.add_argument('--discriminator', required=True,
        type=dynLoadModule("models"),
        help='name of the file containing the discrimintator model definition')
    
    parser.add_argument("split", choices=["train", "test"])

    return parser
