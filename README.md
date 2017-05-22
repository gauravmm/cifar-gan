# Introduction to GANs with CIFAR-100

This is our quick-and-dirty introduction to GANs using TensorFlow and the CIFAR-100 dataset. The aim of this is to figure out the technical challenges in semi-supervised learning and representation learning using GANs before we apply it to a a larger dataset like ImageNet.

## The Goal

Our goal is to develop a semi-supervised classifier using the CIFAR dataset with restricted labels. We will:

1. Train a generator-discriminator pair using the 60,000 CIFAR images _without labels_.
2. Modify the discriminator in some way so that we can use it to extract features from the images. (e.g. we can remove the uppermost layer)
3. Either use the features with a classical learning algorithm (nearest neighbour, random forest, etc.) or append additional layers so that we have a classifier.
4. Train/Fine-tune the classifier on a restricted subset of CIFAR (e.g. 10%) _with labels_.

Part of this project is discovering the best practices and theory behind how each of these steps works. If you have a better idea that doesn't fit in this framework, then go with the idea and disregard the framework.

We are all going to report our experiments and results on the Google Sheet [here](https://docs.google.com/spreadsheets/d/1fVaBiB3TY8EiS3K_oi7miL5MGW4lD_SWG8g-FvbUUq4/edit?usp=sharing).

## Installation

This codebase is targeted at Python 3.5

1. Set up Ubuntu 16.04 (or higher) on a machine.
2. Follow the installation instructions [here](https://www.tensorflow.org/install/install_linux#InstallingNativePip) to install TensorFlow.
3. Install the prerequisites with `sudo -H pip3 install numpy urllib3`.

To get started with development:

1. `git clone git@github.com:gauravmm/cifar-gan.git`
2. `git tag -l` to list the tags
3. `git checkout tags/<tag_name> -b <branch_name>` to create a new branch named `<branch-name>` starting from tag `<tag_name>`.  
    Remember to name the branch after yourself!

To run the code

 1. `./experiment.sh`  
    Output will be logged to `train_logs/` including loss, accuracy, and generated images; and weights will be saved to `weights/`.

## Coding Standards
We will use git for distributed version control. Feel free to submit pull requests with partial/incomplete work. This is research code, and so there are only two coding standards:

1. You _must_ only commit to branches that you create. (Read from everyone's branches, only write to your own.)
2. Any code you use to generate results from _must_ be easily reproducible. This means that:
   1. It should be self-contained in a single commit. (You _should_ indicate that commit hash in the Google spreadsheet we are using to aggregate results.)
   2. We must be able to replicate that on any machine by running `experiment.sh`.
   3. You must set your random seed explicitly so that all runs are identical.

## Bugs & Features

Submit bug reports and feature requests on [the issues page](https://github.com/gauravmm/cifar-gan/issues). Feel free to submit pull requests for bugs or features. DO NOT MERGE INTO PRODUCTION -- submit a pull request and I will handle the merging.
