# Preprocessor that normalizes the input.

import numpy as np
import tensorflow as tf

# Forward must be applied in np
def apply(inp):
    """
    Applies transformations to input and labels, returning a tuple of (preprocessed_input, preprocessed_labels).
    input is an ndarray of shape (batch_size, height, width, channels)
    label is an ndarray of shape (batch_size, num_classes)
    You may create or remove elements from the batch as necessary.
    """
    img, label = inp
    return (img.astype(np.float32)/127.5 - 1.0, label)
    #return (img.astype(np.float32)/255.0, label)

apply_test = apply
apply_train = apply

# Reverse must be implemented in tf
def unapply(inp):
    """
    Inverts the transformation to input, using TensorFlow. This is used to display generated images.
    You may not create or remove elements from the batch.
    """
    return tf.cast((inp + 1) * 127.5, tf.uint8)
    #return tf.cast(inp * 255.0, tf.uint8)