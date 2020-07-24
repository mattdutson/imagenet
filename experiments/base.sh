#!/usr/bin/env bash

#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --job-name=base
#SBATCH --mem-per-gpu=8GB
#SBATCH --output=outputs/base.txt
#SBATCH --partition=batch_default
#SBATCH --time=2-00:00:00

./scripts/train.py base -e 50 -v 2
