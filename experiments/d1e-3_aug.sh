#!/usr/bin/env bash

#SBATCH --cpus-per-task=4
#SBATCH --error=outputs/d1e-3_aug.err
#SBATCH --gres=gpu:2
#SBATCH --job-name=d1e-3_aug
#SBATCH --mem-per-gpu=8GB
#SBATCH --output=outputs/d1e-3_aug.out
#SBATCH --partition=batch_default
#SBATCH --time=5-00:00:00

PYTHONPATH="./:$PYTHONPATH" ./scripts/train.py d1e-3_aug -a -d 1e-3 -e 30 -l 1e-2 -o SGD -v 2
