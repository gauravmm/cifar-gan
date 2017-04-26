# Preprocessor that normalizes the input.

import tensorflow as tf

def apply(inp, label):
    """
    Applies transformations to input and labels, returning a tuple of (preprocessed_input, preprocessed_labels).
    input is an ndarray of shape (batch_size, height, width, channels)
    label is an ndarray of shape (batch_size, num_classes)
    You may create or remove elements from the batch as necessary.
    """

    return (tf.cast(inp, tf.float32)/127.5 - 1.0, label)

apply_test = apply
apply_train = apply

# Reverse 
def unapply(inp):
    """
    Inverts the transformation to input, returning a single ndarray. This is used to display generated images.
    You may not create or remove elements from the batch.
    """
    inp = tf.clip_by_value((inp + 1.0) * 127.5, 0.0, 1.0)
    return tf.cast(inp, tf.uint8)
