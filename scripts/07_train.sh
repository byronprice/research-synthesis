#!/bin/bash
# SLURM job script for fine-tuning on the GPU cluster.
# Edit the SLURM directives for your cluster's partition names and account.
#
# Usage: sbatch scripts/07_train.sh

#SBATCH --job-name=neuro-finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

set -e

echo "=== Job started: $(date) ==="
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Activate your conda/venv environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate neuro-synthesis
# OR:
# source ~/venvs/neuro/bin/activate

# Install training dependencies if not already installed
pip install -q -r requirements/training.txt

# Run training
python src/training/train.py --config configs/training_config.yaml

echo "=== Training complete: $(date) ==="
