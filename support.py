# Data support functions

import numpy as np

from typing import Tuple


# TODO: Support randomization of input
def data_stream(dataset, batch_size : int):
    # The first index of the next batch:
    i = 0 # Type: int
    print(dataset.shape)

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

# Stream of random data
def random_stream(batch_size : int, img_size : Tuple[int, int, int]):
    sz = [batch_size, *img_size]
    while True:
        yield np.random.normal(size=sz)
