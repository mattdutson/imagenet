#!/usr/bin/env bash

#SBATCH --cpus-per-task=4
#SBATCH --error=outputs/d1e-3_s320.err
#SBATCH --gres=gpu:2
#SBATCH --job-name=d1e-3_s320
#SBATCH --mem-per-gpu=8GB
#SBATCH --output=outputs/d1e-3_s320.out
#SBATCH --partition=batch_default
#SBATCH --time=5-00:00:00

PYTHONPATH="./:$PYTHONPATH" ./scripts/train.py d1e-3_s320 -d 1e-3 -e 30 -l 1e-2 -o SGD -v 2 -s 320 320
