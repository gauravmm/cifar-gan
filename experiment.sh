#!/bin/bash
# Suppress info logs
export TF_CPP_MIN_LOG_LEVEL=1
python3 adversarial.py \
    --generator     simple_model \
    --discriminator simple_model \
    --log-interval 100 \
    train \
