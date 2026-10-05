#!/bin/bash

if [ $# != 2 ]; then
    echo "This script must be called with exactly 2 arguments. It was called with $#: $@"
    echo "It is intended to be run by the julia program to re-launch itself."
else
    source ../configuration.conf
    SBATCH_OPTS=(
        --job-name=$1
        --partition=gpu #here
        --qos=gpu #here
        --gres=gpu:a100_80g:1 #here
        --mem=80G
        --time=1-0:00:00
        --output=logs/%x_%j.out
        --error=logs/%x_%j.err
        --mail-user=$NOTIFICATION_EMAIL
        --mail-type=ALL
    )
    shift # removes first argument
    cd $JULIA_FOLDER_PATH
    sbatch ${SBATCH_OPTS[@]} job_sbatch.bash load $@ # to resume
fi
