# CIFAR10 Downloader

import logging
import pickle
import math
import os
import errno
import tarfile
import shutil

import numpy as np

import urllib3

logger = logging.getLogger(__name__)

_shuffle = False
_data_frac = 1.0
def gen(d, batch_size, wrap=True):
    x, y = d
    NUM_CLASS = 10
   
    def _to_1hot(lbl):
        z = np.zeros(shape=(batch_size, NUM_CLASS))
        z[np.arange(batch_size), lbl.flatten()] = 1
        return z

    assert x.shape[0] == y.shape[0]

    if wrap:
        # The first index of the next batch:
        i = 0 # Type: int
        while True:
            j = i + batch_size
            # If we wrap around the back of the dataset:
            if j >= x.shape[0]:
                rv = list(range(i, x.shape[0])) + list(range(0, j - x.shape[0]))
                yield (x[rv,...], _to_1hot(y[rv,...]))
                i = j - x.shape[0]
            else:
                yield (x[i:j,...], _to_1hot(y[i:j,...]))
                i = j
    else:
        i = 0
        j = 0
        while j < (x.shape[0] // batch_size) * batch_size:
            j = i + batch_size
            yield (x[i:j,...], _to_1hot(y[i:j,...]))
            i = j

def configure(props):
    if "shuffle" in props:
        _shuffle = True
        logger.info("Configure: Enabled shuffling data.")

    if "frac" in props:
        f = float(props["frac"])
        if f > 1.0:
            raise RuntimeError("Data fraction must be <= 1.0")
        if f <= 0.0:
            raise RuntimeError("Data fraction must be > 0.0")
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
            x, y = train
            p = np.random.permutation(x.shape[0])
            x = x[p,...]
            y = y[p,...]
            train = (x, y)

        # Restrict the number of training examples
        if _data_frac < 1.0:
            x, y = train
            n = x.shape[0] * _data_frac
            x = x[:n,...]
            y = y[:n,...]
            train = (x, y)

        num_lbl = math.floor(train[0].shape[0] * labelled_fraction)
        # Return (unlabelled, labelled).
        # We strip the labels out and replace them with None, to better data hygene
        return (map(lambda x: (x[0], None), gen(train, batch_size)),
                gen((train[0][:num_lbl,...], train[1][:num_lbl,...]), batch_size))

    elif split =="test":
        test = _get_dataset("test")
        nt = (test[0].shape[0] // batch_size) * batch_size
        return (nt, gen(test, batch_size, False))
    
    assert not "get_data must be called with \"train\" or \"test\"."


def _unpickle_file(filename):
    logger.debug("Loading pickle file: {}".format(filename))

    with open(filename, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    # Reorder the data
    img = data[b'data']
    img = img.reshape([-1, 3, 32, 32])
    img = img.transpose([0, 2, 3, 1])
    # Load labels
    lbl = np.array(data[b'labels'])

    return img, lbl


def _get_dataset(split):
    assert split == "test" or split == "train"
    path = "data"
    dirname = "cifar-10-batches-py"
    data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    if not os.path.exists(os.path.join(path, dirname)):
        # Extract or download data
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        
        file_path = os.path.join(path, data_url.split('/')[-1])
        if not os.path.exists(file_path):
            # Download
            logger.warn("Downloading {}".format(data_url))
            with urllib3.PoolManager().request('GET', data_url, preload_content=False) as r, \
                 open(file_path, 'wb') as w:
                    shutil.copyfileobj(r, w)

        logger.warn("Unpacking {}".format(file_path))
        # Unpack data
        tarfile.open(name=file_path, mode="r:gz").extractall(path)

    # Import the data
    filenames = ["test_batch"] if split == "test" else \
                ["data_batch_{}".format(i) for i in range(1, 6)]
    
    imgs = []
    lbls = []
    for f in filenames:
        img, lbl = _unpickle_file(os.path.join(path, dirname, f))
        imgs.append(img)
        lbls.append(lbl)

    # Now we flatten the arrays
    return np.concatenate(imgs), np.concatenate(lbls)