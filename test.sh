#!/bin/bash

function colorize() (
    set -o pipefail;
    ("$@" 2>&1>&3 | sed -e $'s,\[[0-9:]* INFO.*,\e[93m&\e[m,' -e $'s,\[[0-9:]* INFO @root].*,\e[36m&\e[m,' -e $'s,\[[0-9:]* WARNING.*,\e[91m&\e[m,' -e $'s,\[[0-9:]* ERROR.*,\e[41m&\e[m,' >&2) 3>&1 | sed $'s,.*,\e[m&\e[m,'
)
# Suppress info logs
export CUDA_VISIBLE_DEVICES=1
export TF_CPP_MIN_LOG_LEVEL=2

colorize python3 adversarial.py      \
    --hyperparam    simple           \
    --data          cifar10[shuffle] \
    --preprocessor  normalize        \
    --generator     cifar_resnet     \
    --discriminator cifar_resnet     \
    --log-interval  10               \
    test
