# CIFAR10 Downloader

import glob
import logging
import math
import os
import re

import numpy as np

logger = logging.getLogger(__name__)

NUM_CLASS = 2
CLASS_ID = {"benign": 0, "malignant": 1, "unlabelled": 0}

_shuffle = False
_data_frac = 1.0

def _to_1hot(lbl):
        z = np.zeros(shape=(NUM_CLASS,))
        z[lbl] = 1
        return z

def _list_to_seq(rv):
    # Convert this list-of-(tuple-of-(numpy-arr-image, numpy-arr-label)) to tuple-of-(concat-numpy-arr-image, concat-numpy-arr-label)
    x = np.stack([d[0] for d in rv], axis=0)
    y = np.stack([d[1] for d in rv], axis=0)
    return (x, y)

_load_disk = np.load

def _gen(d, batch_size, wrap=True):
    d = [(_load_disk(f[0]), _to_1hot(f[1])) for f in d]
    logger.info("Loaded {} examples from disk.".format(len(d)))

    l = len(d)
    # The first index of the next batch:
    i = 0 # Type: int
    while True:
        j = i + batch_size
        if j < l: # If we don't wrap around the back of the dataset:
            yield _list_to_seq(d[i:j])
            i = j
        elif wrap: # We only wrap if the wrap is set.
            yield _list_to_seq(d[i:] + d[:j-l])
            i = j-l
        else:
            break # No wrap, and we fell off the end of the list. End the output.

def configure(props):
    if "shuffle" in props:
        _shuffle = True
        logger.info("Configure: Enabled shuffling data.")

    if "frac_unlabelled" in props:
        f = float(props["frac_unlabelled"])
        if f > 1.0:
            raise RuntimeError("Fraction of unlabelled data must be <= 1.0")
        if f <= 0.0:
            raise RuntimeError("Fraction of unlabelled data must be > 0.0")
        _data_frac = f
        logger.info("Configure: Training set fraction is {}".format(_data_frac))

def get_data(split, batch_size, labelled_fraction=1.0):
    """The provider function in a dataset.

    This function provides the training, development, and test sets as necessary.

    Args:
        split      (str): "train", "develop", or "test"; to indicate the appropriate split. "train" provides an infinite
                          generator that constantly loops over the input data; the other two end after all data is
                          consumed. Note: "develop" is not implemented.
        batch_size (int): The number of images to provide in each batch. Finite generators will discard partial batches.
        labelled_fraction (float): The fraction of "train" data to provide with labels.

    Returns:
        tuple: (if "train")
            generator: (infinite)
                tuple: 
                    (batch_size, 32, 32, 3) Unlabelled image
                    (batch_size, 10, 1)     Zeros
            generator: (infinite)
                tuple: 
                    ndarray: (batch_size, 32, 32, 3) Labelled image
                    ndarray: (batch_size, 10, 1)     1-hot label encoding
        tuple: (otherwise)
            int: number of examples in the generator
            generator: (finite)
                tuple: 
                    ndarray: (batch_size, 32, 32, 3) Labelled image
                    ndarray: (batch_size, 10, 1)     1-hot label encoding
    """
    
    if split == "train":
        train = _get_dataset("train")
        # Shuffle the training data
        if _shuffle:
            train["labelled"] = train["labelled"][np.random.permutation(len(train["labelled"]))]
            train["unlabelled"] = train["labelled"][np.random.permutation(len(train["unlabelled"]))]

        # Restrict the number of unlabelled examples
        if _data_frac < 1.0:
            n = len(train["unlabelled"]).shape[0] * _data_frac
            train["unlabelled"] = train["unlabelled"][:n]

        num_lbl = math.floor(len(train["labelled"]) * labelled_fraction)
        
        # Return (unlabelled, labelled).
        # We strip the labels out and replace them with None, for better data hygene
        return (map(lambda x: (x[0], None), _gen(train["unlabelled"], batch_size)),
                _gen(train["labelled"][:num_lbl], batch_size))

    elif split =="test":
        test = _get_dataset("test")
        nt = (len(test) // batch_size) * batch_size
        return (nt, _gen(test, batch_size, False))
    
    assert not "get_data must be called with \"train\" or \"test\"."


def _get_dataset(split):
    assert split == "test" or split == "train"
    prefix = "data/nodules/" + split
    path = prefix + "/*/*.npy"
    regex = re.compile("^" + prefix + "/(benign|malignant|unlabelled)/([ -~]+).npy$")

    dataset = {"labelled": [], "unlabelled": []}

    count = {"benign": 0, "malignant": 0, "unlabelled": 0}

    for p in glob.glob(path):
        clas, name = regex.match(p).groups() # Properties: class, name:
        count[clas] += 1
        lbld = "unlabelled" if clas == "unlabelled" else "labelled"
        dataset[lbld].append((p, CLASS_ID[clas]))

    count["all_labelled"] = count["benign"] + count["malignant"]

    logger.info("Loaded metadata: {}".format(", ".join("{}: {}".format(k, v) for k, v in count.items())))

    return dataset if split == "train" else dataset["labelled"]
