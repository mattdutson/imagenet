#!/usr/bin/env bash

#SBATCH --cpus-per-task=4
#SBATCH --error=outputs/d1e-3_aug_p2.err
#SBATCH --gres=gpu:2
#SBATCH --job-name=d1e-3_aug_p2
#SBATCH --mem-per-gpu=8GB
#SBATCH --output=outputs/d1e-3_aug_p2.out
#SBATCH --partition=batch_default
#SBATCH --time=5-00:00:00

# This is part 2 of d1e-3_aug
# Trains for 30 more epochs at a 10x lower LR

PYTHONPATH="./:$PYTHONPATH" ./scripts/train.py d1e-3_aug_p2 -a -d 1e-3 -e 30 -l 1e-3 -o SGD -v 2 \
    -w models/d1e-3_aug.h5
