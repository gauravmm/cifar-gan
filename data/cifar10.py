# CIFAR10 Downloader

from keras.datasets import cifar10
import math

def gen(d, batch_size):
    x, y = d
    NUM_CLASS = 10

    def _to_1hot(lbl):
        z = np.zeros(size=(batch_size, NUM_CLASS))
        z[:,lbl] = 1
        return z

    assert x.shape[0] == y.shape[0]

    # The first index of the next batch:
    i = 0 # Type: int
    while True:
        j = i + batch_size
        # If we wrap around the back of the dataset:
        if j >= dataset.shape[0]:
            rv = list(range(i, dataset.shape[0])) + list(range(0, j - dataset.shape[0]))
            yield (x[rv,...], _to_1hot(y[rv,...]))
            i = j - dataset.shape[0]
        else:
            yield (x[i:j,...], _to_1hot(y[i:j,...]))
            i = j


def get_data(split, batch_size, labelled_fraction=1.0):
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train, test = cifar10.load_data()
    p = np.random.permutation(train.shape[0])
    # Shuffle the data
    train = train[p,...]
    test = test[p,...]
    
    if split == "train":
        # Return (unlabelled, labelled).
        # We strip the labels out and replace them with None, to better data hygene
        return (map(lambda x: (x[0], None), gen(train, batch_size)),
                gen(train[:math.floor(train.shape[0] * labelled_fraction),...], batch_size))
    elif split =="test":
        return gen(test, batch_size)
    
    assert not "get_data must be called with \"train\" or \"test\"."
