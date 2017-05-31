# How To Use

This document explains how to set up and run experiments 

## Model

### Batch Normalization

You must use the layer at `tf.layers.batch_normalization`, not the `contrib` layer. Check your code carefully, because
the contrib layer was very popular until recently.

### Reuse

The discriminator is instantiated twice, once for 'real' and once for 'fake' data. You _should_ share all convolutional
and dense weights and biases between the real and fake models, and you _should not_ share batch normalization parameters
between these nodes.

Operations defined in the discriminator automatically have their parameters shared across the fake and read instances.
In order to not share the fake and real models, we explicitly 
