#!/bin/bash

EXP_DIR="experiments"
EXP_LOG="$EXP_DIR/log.txt"

if [ ! -d .git ]; then
    echo "You must run this while in the same folder as the repository."
    echo "Exiting..."
    exit 1
fi

if [ ! -d "$EXP_DIR" ]; then
    mkdir "$EXP_DIR"
    touch "$EXP_LOG"
else
    echo "$(date) Started experiments" > "$EXP_LOG"
fi

rm -r "train_logs"

for commithash in "$@"
do
    if [ ! -f experiment.sh ]; then
        echo "$(date) Skipping $commithash, experiment.sh not found." >> "$EXP_LOG"
        continue
    fi

    expdir="$EXP_DIR/$commithash"

    if [ -e "$expdir" ]; then
        echo "$(date) Skipping $commithash, $expdir already exists." >> "$EXP_LOG"
        continue
    fi

    echo "$(date) Starting $commithash" >> "$EXP_LOG"
    mkdir "$expdir"
    git reset --hard HEAD
    git pull --all
    if ! git checkout $commithash ; then
        echo "$(date) Skipping $commithash, cannot checkout." >> "$EXP_LOG"
        continue
    fi

    # Run the experiment
    "./experiment.sh"

    echo "$(date) Moving output folders." >> "$EXP_LOG"
    mv "train_logs" "$commithash/"

    echo "$(date) Done!" >> "$EXP_LOG"
done
