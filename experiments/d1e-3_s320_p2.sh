#!/usr/bin/env bash

#SBATCH --cpus-per-task=4
#SBATCH --error=outputs/d1e-3_s320_p2.err
#SBATCH --gres=gpu:2
#SBATCH --job-name=d1e-3_s320_p2
#SBATCH --mem-per-gpu=8GB
#SBATCH --output=outputs/d1e-3_s320_p2.out
#SBATCH --partition=batch_default
#SBATCH --time=5-00:00:00

# This is part 2 of d1e-3_s320
# Trains for 30 more epochs at a 10x lower LR

PYTHONPATH="./:$PYTHONPATH" ./scripts/train.py d1e-3_s320_p2 -d 1e-3 -e 30 -l 1e-3 -o SGD -v 2 -s 320 320 \
    -w models/d1e-3_s320.h5
