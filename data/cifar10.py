# CIFAR10 Downloader

from keras.datasets import cifar10

def gen(d, batch_size):
    x, y = d

    assert x.shape[0] == y.shape[0]

    # The first index of the next batch:
    i = 0 # Type: int
    while True:
        j = i + batch_size
        # If we wrap around the back of the dataset:
        if j >= dataset.shape[0]:
            rv = list(range(i, dataset.shape[0])) + list(range(0, j - dataset.shape[0]))
            yield (x[rv,...], y[rv,...])
            i = j - dataset.shape[0]
        else:
            yield (x[i:j,...], y[i:j,...])
            i = j


def get_data(split):
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train, test = cifar10.load_data()
    
    if split == "train":
        return gen(train)
    elif split =="test":
        return gen(test)
    
    assert not "get_data must be called with \"train\" or \"test\"."
