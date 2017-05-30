# Preprocessor that normalizes the input.

import numpy as np
import tensorflow as tf
import logging

logger = logging.getLogger(__file__)

# Forward must be applied in np
def apply(inp):
    """
    Applies transformations to input and labels, returning a tuple of (preprocessed_input, preprocessed_labels).
    input is an ndarray of shape (batch_size, height, width, channels)
    label is an ndarray of shape (batch_size, num_classes)
    You may create or remove elements from the batch as necessary.
    """
    x, y = inp 
    return (np.expand_dims(x, 4), y)  # Add an extra dimension for channels.

apply_test = apply
apply_train = apply

# Reverse must be implemented in tf
def unapply(inp):
    """
    Inverts the transformation to input, using TensorFlow. This is used to display generated images.
    You may not create or remove elements from the batch.
    """
    # We convert the 3d volume to a 2d image.

    inp = tf.reshape(inp, [-1, 32, 32*32, 1])
    
    return tf.cast(inp * 255.0, tf.uint8)
