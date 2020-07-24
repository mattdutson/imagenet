#!/usr/bin/env bash

#SBATCH --cpus-per-task=4
#SBATCH --error=outputs/r1e-3_p2.err
#SBATCH --gres=gpu:2
#SBATCH --job-name=r1e-3_p2
#SBATCH --mem-per-gpu=8GB
#SBATCH --output=outputs/r1e-3_p2.out
#SBATCH --partition=batch_default
#SBATCH --time=5-00:00:00

# This is part 2 of r1e-3.sh
# Trains for 30 more epochs at the same LR

PYTHONPATH="./:$PYTHONPATH" ./scripts/train.py r1e-3_p2 -e 30 -l 1e-3 -o RMSprop -v 2 \
    -w models/r1e-3.h5
