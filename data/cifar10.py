# CIFAR10 Downloader

from keras.datasets import cifar10
import numpy as np
import math

def gen(d, batch_size):
    x, y = d
    NUM_CLASS = 10

    # Randomize the order
    p = np.random.permutation(x.shape[0])
    x = x[p,...]
    y = y[p,...]
    
    def _to_1hot(lbl):
        z = np.zeros(shape=(batch_size, NUM_CLASS))
        z[:,lbl] = 1
        return z

    assert x.shape[0] == y.shape[0]

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


def get_data(split, batch_size, labelled_fraction=1.0):
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
        return gen(test, batch_size)
    
    assert not "get_data must be called with \"train\" or \"test\"."
