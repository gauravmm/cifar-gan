# Preprocessor that normalizes the input.

import numpy as np

def apply(inp, label):
    """
    Applies transformations to input and labels, returning a tuple of (preprocessed_input, preprocessed_labels).
    input is an ndarray of shape (batch_size, height, width, channels)
    label is an ndarray of shape (batch_size, num_classes)
    You may create or remove elements from the batch as necessary.
    """

    return (inp.astype(np.float32)/127.5 - 0.5, label)

apply_test = apply
apply_train = apply

# Reverse 
def unapply(inp):
    """
    Inverts the transformation to input, returning a single ndarray. This is used to display generated images.
    You may not create or remove elements from the batch.
    """
    inp = (inp + 0.5) * 127.5
    inp[inp>255] = 255
    inp[inp<0] = 0
    return inp.astype(np.uint8)
