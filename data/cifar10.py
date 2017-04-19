# CIFAR10 Downloader

from keras.datasets import cifar10
import numpy as np
import math

def gen(d, batch_size, wrap=True):
    x, y = d
    NUM_CLASS = 10

    # Randomize the order
    p = np.random.permutation(x.shape[0])
    x = x[p,...]
    y = y[p,...]
    
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

    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train, test = cifar10.load_data()
    # Shuffle the training data
    num_lbl = math.floor(train[0].shape[0] * labelled_fraction)

    if split == "train":
        # Return (unlabelled, labelled).
        # We strip the labels out and replace them with None, to better data hygene
        return (map(lambda x: (x[0], None), gen(train, batch_size)),
                gen((train[0][:num_lbl,...], train[1][:num_lbl,...]), batch_size))
    elif split =="test":
        nt = (test[0].shape[0] // batch_size) * batch_size
        return (nt, gen(test, batch_size, False))
    
    assert not "get_data must be called with \"train\" or \"test\"."
