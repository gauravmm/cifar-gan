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

# Produces a stream of random data
def random_stream(batch_size : int, img_size : Tuple[int, int, int]):
    sz = [batch_size, *img_size]
    while True:
        yield np.random.normal(size=sz)


def get_latest_blob(blob):
    """
    Returns the file that matches blob (with a single wildcard), that has the highest numeric value in the wildcard.
    """
    assert len(filter(lambda x: x == "*", blob)) == 1 # There should only be one wildcard in blob
    
    blobs = glob.glob(blob)
    assert len(blobs) # There should be at least one matchs
    
    ltrunc = blob.index("*")           # Number of left characters to remove
    rtrunc = -(len(blob) - ltrunc + 1) # Number of right characters to remove
    
    # Get the indices hidden behind the wildcard
    idx = [int(b[ltrunc:rtrunc]) for b in blobs
    return next(sorted(zip(idx, blobs), reverse=True))
    

