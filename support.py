# Data support functions

import argparse
import glob
import importlib
import itertools
import logging
import sys

import numpy as np

import config
from typing import Tuple

logger = logging.getLogger()

#
# Data
#

# TODO: Support randomization of input
def data_stream(dataset, batch_size : int):
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
def random_stream(batch_size : int, img_size : Tuple[int, int, int]):
    sz = [batch_size, *img_size]
    while True:
        yield np.random.normal(size=sz)


#
# Filesystem
#

def get_latest_blob(blob):
    """
    Returns the file that matches blob (with a single wildcard), that has the highest numeric value in the wildcard.
    """
    assert len(filter(lambda x: x == "*", blob)) == 1 # There should only be one wildcard in blob
    
    blobs = glob.glob(blob)
    assert len(blobs) # There should be at least one matchs
    
    ltrunc = blob.index("*")           # Number of left characters to remove
    rtrunc = -(len(blob) - ltrunc + 1) # Number of right characters to remove
    
    # Get the indices hidden behind the wildcard
    idx = [int(b[ltrunc:rtrunc]) for b in blobs]
    return next(sorted(zip(idx, blobs), reverse=True))

def resume(args, gen_model, dis_model):
    try:
        # Load files as necessary
        gen_num, gen_fn = get_latest_glob(config.get_filename('weight', args, 'gen'))
        dis_num, dis_fn = get_latest_glob(config.get_filename('weight', args, 'dis'))

        # Check if the files are from the same batch.
        assert gen_num == dis_num
        
        gen_model.load_weights(gen_fn, by_name=True)
        logger.info("Loaded generator weights from {}".format(gen_fn))
        dis_model.load_weights(dis_fn, by_name=True)
        logger.info("Loaded discriminator weights from {}".format(gen_fn))
        return gen_num
    except:
        logger.warn("Caught exception {}s".format(sys.exc_info()[0]))
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
        help='the name of files with image preprocessing instructions in the preprocessor package; may be applied in any order')
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
