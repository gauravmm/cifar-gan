#!/bin/bash
# Suppress info logs
export TF_CPP_MIN_LOG_LEVEL=2
python3 adversarial.py           \
    --preprocessor  normalize    \
    --generator     simple_model \
    --discriminator simple_model \
    --log-interval 50            \
    --batches 2500               \
    --image-columns 4            \
    train
