#!/bin/python3

"""
Main CIFAR-GAN file.
Handles the loading of data from ./data, models from ./models, training, and testing.

Modified from TensorFlow-Slim examples and https://github.com/wayaai/GAN-Sandbox/
"""

import argparse, importlib, sys

def main(args):
    print(args)

    


#
# Command-line handlers
#

def dynLoadModule(pkg):
    """
    Used to dynamically load modules in commandline options.
    """
    return lambda modname: importlib.import_module(pkg + "." + modname, package=".")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test GAN models on data.')

    parser_g1 = parser.add_mutually_exclusive_group(required=True)
    parser_g1.add_argument('--train', action='store_const', dest='split', const='train', default='')
    parser_g1.add_argument('--test', action='store_const', dest='split', const='test', default='')

    parser.add_argument('--data', metavar='D', default="cifar10", type=dynLoadModule("data"),
                    help='the name of a tf.slim dataset reader in the data package')
    parser.add_argument('--generator', metavar='G', type=dynLoadModule("models"),
                    help='name of the module containing the generator model definition')
    parser.add_argument('--discriminator', metavar='S', type=dynLoadModule("models"),
                    help='name of the module containing the discrimintator model definition')
    args = parser.parse_args()

    main(args)
