# Preprocessor that normalizes the input.

import numpy as np
import tensorflow as tf
import logging

logger = logging.getLogger(__file__)

NUM_COL = 4

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

    images = tf.unstack(inp, axis=3)
    columns = []

    # logger.info(inp.get_shape())

    # assert (len(images) % NUM_COL) == 0
    
    for i in range(0, len(images), NUM_COL):
        columns.append(tf.concat(images[i:(i+NUM_COL)], axis=0))
    
    rv = tf.concat(columns, axis=1)
    
    return tf.cast(rv * 255.0, tf.uint8)
