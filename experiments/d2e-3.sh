#!/usr/bin/env bash

#SBATCH --cpus-per-task=4
#SBATCH --error=outputs/d2e-3.err
#SBATCH --gres=gpu:2
#SBATCH --job-name=d2e-3
#SBATCH --mem-per-gpu=8GB
#SBATCH --output=outputs/d2e-3.out
#SBATCH --partition=batch_default
#SBATCH --time=3-00:00:00

PYTHONPATH="./:$PYTHONPATH" ./scripts/train.py d2e-3 -d 2e-3 -e 30 -l 1e-2 -o SGD -v 2
