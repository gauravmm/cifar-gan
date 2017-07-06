# Data support functions

import argparse
import errno
import functools
import glob
import importlib
import inspect
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
# Batch Normalization Support
#

class BatchNormLayerFactory(object):
    def get_layers(self, prefixes, arg):
        """
        Assemble multiple batch-norm layers, all sharing the same shape as necessary to work with arg.
        The return value is a list-of-lambdas, each lambda is a tensorflow op that can be composed onto any input node.
        The layers are returned with identical starting states, and each with the prefix corresponding to the entry in
        prefixes.
        """
        args, kwargs = arg

        return None

class TFMultiFactoryEntry(object):
    def __init__(self, func, prefixes, name, arg, defn):
        self.vals = []
        args, kwargs = arg

        for p in prefixes:
            kwargs['name'] = p + name
            self.vals.append(func(*args, **kwargs))
        
        self.defn = defn
    
    def getDefinitionPlace(self):
        return self.defn
    
    def apply(self, i):
        return self.vals[i]


class TFMultiFactory(object):
    _multifactory_idx = 0
    def __init__(self, func, prefixes=("a_", "b_"), scope=None):
        assert len(prefixes) == 2
        _multifactory_idx = TFMultiFactory._multifactory_idx
        TFMultiFactory._multifactory_idx = _multifactory_idx + 1

        self.idx = 0
        self.allowCreation = True
        self.func = func
        self.prefixes = prefixes
        self.scope = scope
        self.maps = {}
        self.name_prefix = "multifactory" + str(_multifactory_idx)
        self.logger = logging.getLogger('TFMultiFactory:' + str(_multifactory_idx))
        
        # Reset the name generator
        self._reset_name_gen()
    
    def _reset_name_gen(self):
        self.name_gen = (self.name_prefix + "_" + str(i) for i in itertools.count())

    def reuse(self, i=None):
        self.allowCreation = False
        if i is None:
            self.idx += 1
        else:
            self.idx = i
        self._reset_name_gen()

    
    def __call__(self, *args, **kwargs):
        st = inspect.stack()
        defn = "({}){}:{} ".format(st[1].function, st[1].filename, st[1].lineno)

        # Extract the name from the args.
        try:
            name = kwargs['name']
        except KeyError:
            name = next(self.name_gen)
            self.logger.warning("You should specify a name for the layer at {}, assigning \"{}\".".format(defn, name))

        if self.scope.reuse:
            # If this is the first time we're allowing reuse:
            if self.allowCreation:
                self.logger.info("Enabled reuse")
                # Check that we only have two prefixes. We insist on using reuse explicitly if we have more than two
                # separate uses.
                if len(self.prefixes) != 2:
                    self.logger.error("Attempting to implicitly switch scope to reuse when more than two uses are expected! You must explicitly call reuse() before your call at {}.".format(defn))
                    raise AssertionError()
                self.reuse()

            if name not in self.maps:
                self.logger.error("All TFMultiFactory objects must be defined before .reuse() is called or the scope is switched to reuse-mode! Call at {}.".format(defn))
                raise AssertionError()

            # Now we retrieve the corresponding entry:
            self.logger.debug("Round {}, matching original {} -> reuse -> {}".format(self.idx, self.maps[name].getDefinitionPlace(), defn))

        else:            
            if name in self.maps:
                self.logger.error("All TFMultiFactory objects must have a unique name! \"{}\" is repeated. First use in {}.".format(name, self.maps[name].getDefinitionPlace()))
                raise AssertionError()

            # Add a new entry to the maps:
            self.maps[name] = TFMultiFactoryEntry(self.func, self.prefixes, name, (args, kwargs), defn)
        
        logger.info(self.maps.__repr__())

        return self.maps[name].apply(self.idx)

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

# A generator that enforces the batch-size of the input. Used to feed TensorFlow the right amount of data even with data
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
        for i in range(batch_size):
            z[i, q[i]] = 1
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
        help='name of the file containing the discriminator model definition')
    parser.add_argument('--only-classifier-after', default=-1, type=int,
        help='only train the classifier, starting from this batch number')
    
    parser.add_argument("split", choices=["train", "test"])

    return parser
