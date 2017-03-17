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
from typing import Tuple

logger = logging.getLogger()

#
# Data
#

class Data(object):
    def __init__(self, args):
        train_data, train_labels = args.data.get_data("train")
        logger.info("Data loaded from disk.")

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

        # Use to label a batch as real
        self.label_real = np.array([0] * args.hyperparam.batch_size)  # Label to train discriminator on real data
        # Use to label a batch as fake
        self.label_fake = np.array([1] * args.hyperparam.batch_size)  # Label to train discriminator on generated data
    

# TODO: Support preprocessors.
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
    sz = [batch_size, *img_size]
    while True:
        yield np.random.normal(size=sz)


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
    rtrunc = -(len(blob) - ltrunc + 1) # Number of right characters to remove
    
    # Get the indices hidden behind the wildcard
    idx = [int(b[ltrunc:rtrunc]) for b in blobs]
    return next(sorted(zip(idx, blobs), reverse=True))

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
        logger.info("Loaded discriminator weights from {}".format(gen_fn))

        return gen_num

    except Exception as e:
        logger.warn("Caught exception: {}".format(e))
        return None

def clear(args):
    # Delete old weight checkpoints
    for f in itertools.chain(glob.glob(config.get_filename('weight', args)),
                             glob.glob(config.get_filename('image',  args))):
        logger.debug("Deleting file {}".format(f))
        os.remove(f)
    logger.info("Deleted all saved weights and images for generator \"{}\" and discriminator \"{}\".".format(args.generator.NAME, args.discriminator.NAME))

    with open(config.get_filename('csv', args), 'w') as csvfile:
        print("time, batch, composite loss, discriminator+real loss, discriminator+fake loss", file=csvfile)
        logger.debug("Wrote headers to CSV file {}".format(csvfile.name))

    return 0

#
# CLI
#

def dynLoadModule(pkg):
    # Used to dynamically load modules in commandline options.
    return lambda modname: importlib.import_module(pkg + "." + modname, package=".")

def argparser():
    parser = argparse.ArgumentParser(description='Train and test GAN models on data.')

    parser.add_argument('--preprocessor', nargs="*", default=[],
        type=dynLoadModule("preprocessor"),
        help='the name of files with image preprocessing instructions in the preprocessor package; applied in left-to-right order')
    parser.add_argument('--log-interval', default=config.LOG_INTERVAL_DEFAULT, type=int,
        help="the number of batches between reporting results and saving weights")
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
