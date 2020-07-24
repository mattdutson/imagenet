#!/usr/bin/env bash

#SBATCH --cpus-per-task=4
#SBATCH --error=outputs/s1e-2.err
#SBATCH --gres=gpu:2
#SBATCH --job-name=s1e-2
#SBATCH --mem-per-gpu=8GB
#SBATCH --output=outputs/s1e-2.out
#SBATCH --partition=batch_default
#SBATCH --time=5-00:00:00

PYTHONPATH="./:$PYTHONPATH" ./scripts/train.py s1e-2 -e 30 -l 1e-2 -o SGD -v 2
