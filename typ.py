# Type and class definitions

import collections

Hyperparameters = collections.namedtuple('Hyperparameters', 'optimizer halt_batches batch_size discriminator_per_step generator_per_step')