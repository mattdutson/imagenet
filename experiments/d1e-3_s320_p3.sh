#!/usr/bin/env bash

#SBATCH --cpus-per-task=4
#SBATCH --error=outputs/d1e-3_s320_p3.err
#SBATCH --gres=gpu:2
#SBATCH --job-name=d1e-3_s320_p3
#SBATCH --mem-per-gpu=8GB
#SBATCH --output=outputs/d1e-3_s320_p3.out
#SBATCH --partition=batch_default
#SBATCH --time=5-00:00:00

# This is part 3 of d1e-3_s320
# Trains for 30 more epochs at a 100x lower LR

PYTHONPATH="./:$PYTHONPATH" ./scripts/train.py d1e-3_s320_p3 -d 1e-3 -e 30 -l 1e-4 -o SGD -v 2 -s 320 320 \
    -w models/d1e-3_s320_p2.h5
