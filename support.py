# Data support functions

import argparse
import errno
import functools
import glob
import importlib
import itertools
import logging
import os
import re
import sys

import numpy as np

import config
from config import IMAGE_GUTTER
from typing import Tuple

logger = logging.getLogger()

#
# Math
#

class MovingAverage(object):
    def __init__(self, period):
        self.period = period
        self.i = 0
        self.num = 0
        self.vals = [0 for _ in range(period)]
        self.mean = 0

    def push(self, val):
        self.num += 1
        self.mean += (val - self.vals[self.i])/min(self.period, self.num)
        self.vals[self.i] = val
        self.i += 1
        if self.i >= self.period:
            self.i = 0
            # Recalculate mean to prevent error drift.
            self.mean = np.mean(self.vals)

    def get(self):
        return self.mean

#
# Data
#

Y_REAL = 0
Y_FAKE = 1

class Preprocessor(object):
    def __init__(self, args):
        logger = logging.getLogger("preprocessor")
        # Assemble the preprocessor:
        # We apply all the preprocessors in order to get a generator that automatically applies preprocessing.
        self.unapply = functools.reduce(lambda f, g: lambda x: g(f(x)), [p.unapply for p in reversed(args.preprocessor)], lambda x: x)
        
        applyfunc = lambda h: functools.reduce(lambda f, g: lambda x: g(f(x)), [h(p) for p in args.preprocessor], lambda x: x)
        self.apply_train = applyfunc(lambda p: p.apply_train)
        self.apply_test = applyfunc(lambda p: p.apply_test)

        logger.info("Loaded preprocessors: {}.".format(
                        " -> ".join([x.__name__[13:] if x.__name__[:13] == "preprocessor." else x.__name__ for x in args.preprocessor])))


class TrainData(object):
    def __init__(self, args, preproc):
        logger = logging.getLogger("traindata")

        unlabelled, labelled = args.data.get_data("train", args.hyperparam.BATCH_SIZE, labelled_fraction=args.hyperparam.LABELLED_FRACTION)
        logger.info("Training data loaded from disk.")

        unlabelled = map(preproc.apply_train, unlabelled)
        labelled = map(preproc.apply_train, labelled)
        logger.info("Applied training preprocessor.")

        self.rand_vec        = _random_stream(args.hyperparam.BATCH_SIZE, args.hyperparam.SEED_DIM)
        self.rand_label_vec  = _random_1hot_stream(args.hyperparam.BATCH_SIZE, args.hyperparam.NUM_CLASSES)
        # Present images them in chunks of exactly batch-size:
        self.unlabelled      = _image_stream_batch(unlabelled, args.hyperparam.BATCH_SIZE)
        self.labelled        = _image_stream_batch(labelled, args.hyperparam.BATCH_SIZE)

        # Use to label a discriminator batch as real
        self._label_dis_real = map(lambda a, b: a + b,
                            _value_stream(args.hyperparam.BATCH_SIZE, Y_REAL),
                            _function_stream(lambda: args.hyperparam.label_smoothing(True, args.hyperparam.BATCH_SIZE)))
        # Use to label a discriminator batch as fake
        self._label_dis_fake = map(lambda a, b: a + b,
                            _value_stream(args.hyperparam.BATCH_SIZE, Y_FAKE),
                            _function_stream(lambda: args.hyperparam.label_smoothing(False, args.hyperparam.BATCH_SIZE)))
        # Random flipping support
        self.label_dis_real = _selection_stream([args.hyperparam.label_flipping_prob], self._label_dis_fake, self._label_dis_real)
        self.label_dis_fake = _selection_stream([args.hyperparam.label_flipping_prob], self._label_dis_real, self._label_dis_fake)
        # Use to label a generator batch as real
        self.label_gen_real = _value_stream(args.hyperparam.BATCH_SIZE, Y_REAL)

class TestData(object):
    def __init__(self, args, preproc):
        num, labelled = args.data.get_data("test", args.hyperparam.BATCH_SIZE)
        logger.info("Training data loaded from disk.")

        self.labelled = map(preproc.apply_test, labelled)
        logger.info("Applied test preprocessor.")
        
        self.num_labelled = num
        self.label_dis_real = _value_stream(args.hyperparam.BATCH_SIZE, Y_REAL)

#
# Accuracy metric
#

def _accuracy_metric(value):
    # We evaluate k == value, but with only tensor operations.
    return lambda k: 1 - K.abs(K.clip(K.round(k), 0., 1.) - value)

def label_real(y_true, y_pred):
    assert Y_REAL == 0
    return 1 - K.round(K.clip(y_pred, 0., 1.))

def label_fake(y_true, y_pred):
    assert Y_FAKE == 1
    return K.round(K.clip(y_pred, 0., 1.))

METRICS = [label_real, label_fake, 'accuracy']

def get_metric_names(com_model, dis_model_labelled, dis_model_unlabelled, gen_model):
    logger = logging.getLogger("metric_names")

    # Keras overwrites the names of metrics, so here we check that their order is as expected before creating a custom
    # name array.
    logger.debug("Metrics for gen_model: {}".format(gen_model.metrics_names))
    assert gen_model.metrics_names == ['loss']
    logger.debug("Metrics for dis_model_labelled: {}".format(dis_model_labelled.metrics_names))
    assert dis_model_labelled.metrics_names == ['loss', 'discriminator_loss', 'classifier_loss', 'discriminator_label_real', 'discriminator_label_fake', 'discriminator_acc', 'classifier_label_real', 'classifier_label_fake', 'classifier_acc']
    logger.debug("Metrics for dis_model_unlabelled: {}".format(dis_model_unlabelled.metrics_names))
    assert dis_model_unlabelled.metrics_names == ['loss', 'discriminator_loss', 'classifier_loss', 'discriminator_label_real', 'discriminator_label_fake', 'discriminator_acc', 'classifier_label_real', 'classifier_label_fake', 'classifier_acc']
    logger.debug("Metrics for com_model: {}".format(com_model.metrics_names))
    assert com_model.metrics_names == ['loss', 'model_discriminator_unlabelled_loss', 'model_discriminator_unlabelled_loss', 'model_discriminator_unlabelled_label_real', 'model_discriminator_unlabelled_label_fake', 'model_discriminator_unlabelled_acc', 'model_discriminator_unlabelled_label_real', 'model_discriminator_unlabelled_label_fake', 'model_discriminator_unlabelled_acc']
    
    # Custom name array.
    return ['loss', 'discriminator_loss', 'classifier_loss', 'discriminator_label_real', 'discriminator_label_fake', 'discriminator_acc', 'classifier_label_real', 'classifier_label_fake', 'classifier_acc']

# TODO: Support reading test data.

# A generator that enforces the batch-size of the input. Used to feed keras the right amount of data even with data 
# augmentation increasing the batch size.
def _image_stream_batch(itr, batch_size):
    rx, ry = next(itr)
    if ry is None:
        while True:
            while rx.shape[0] < batch_size:
                ax, ay = next(itr)
                rx = np.concatenate((rx, ax))
            yield (rx[:batch_size,...], None)
            rx = rx[batch_size:,...]
    else:
        while True:
            assert rx.shape[0] == ry.shape[0]
            while rx.shape[0] < batch_size:
                ax, ay = next(itr)
                rx = np.concatenate((rx, ax))
                ry = np.concatenate((ry, ay))
            yield (rx[:batch_size,...], ry[:batch_size,...])
            rx = rx[batch_size:,...]
            ry = ry[batch_size:,...]

def _random_1hot_stream(batch_size : int, num_class):
    while True:
        z = np.zeros((batch_size, num_class))
        q = np.random.randint(num_class, size=(batch_size,))
        z[:,q] = 1
        yield z

# Produces a stream of random data
def _random_stream(batch_size : int, img_size):
    sz = [batch_size] + list(img_size)
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

def _make_path(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

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

        return gen_num

    except Exception as e:
        logger.warn("Exception: {}".format(e))
        logger.debug(sys.exc_info())
        return None

def clear(args):
    _make_path(config.get_filename('.', args))

    # Delete old weight checkpoints
    for f in itertools.chain(glob.glob(os.path.join(config.get_filename('.', args), "*"))):
        logger.debug("Deleting file {}".format(f))
        os.remove(f)
    logger.info("Deleted all saved weights and images for generator \"{}\" and discriminator \"{}\".".format(args.generator.NAME, args.discriminator.NAME))

    return 0

#
# CLI
#

def dynLoadModule(pkg):
    # Used to dynamically load modules in commandline options.
    def _loadActual(modnamefull):
        m = re.fullmatch('(\w+)(\[((\w+(=[\w\.\+\-]+)?,?)*)\])?', modnamefull)
        if m is None:
            raise RuntimeError("Incorrect import.")
        
        modname = m.group(1)
        l = importlib.import_module(pkg + "." + modname, package="")
        
        props = m.group(3)
        if props is not None:
            if hasattr(l, 'configure'):
                props = (v.split('=', 1) for v in props.split(","))
                props = {v[0]: (v[1] if len(v) == 2 else True) for v in props}
                logger.debug("Configuring {}.{} with options: {}".format(pkg, modname, str(props)))
                l.configure(props)
            else:
                raise RuntimeError('Options passed to item that does not support options.')

        return l
    
    return _loadActual

def argparser():
    parser = argparse.ArgumentParser(description='Train and test GAN models on data.')

    parser.add_argument('--preprocessor', nargs="*", default=[],
        type=dynLoadModule("preprocessor"),
        help='the name of files with image preprocessing instructions in the preprocessor package; applied in left-to-right order')
    parser.add_argument('--log-interval', default=config.LOG_INTERVAL_DEFAULT, type=int,
        help="the interval, in seconds, at which weights are saved")
    parser.add_argument('--batches', default=config.NUM_BATCHES_DEFAULT, type=int,
        help='the last batch number to process')

    parser.add_argument('--data',
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
