#!/bin/bash
#SBATCH --job-name=eval_baseline
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/eval_baseline_%j.log

module load python/3.11.11-5e66 cuda/12.9.0-cinr
source ~/piano_env/bin/activate
export OSCAR_SCRATCH=/oscar/scratch/$USER/piano_transcription
python train.py --eval_only $OSCAR_SCRATCH/checkpoints/best_so_far.pt
