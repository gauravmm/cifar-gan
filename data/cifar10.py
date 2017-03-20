# CIFAR10 Downloader

from keras.datasets import cifar10

def get_data(split):
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train, test = cifar10.load_data()
    
    if split == "train":
        return train
    elif split =="test":
        return test
    
    assert not "get_data must be called with \"train\" or \"test\"."
