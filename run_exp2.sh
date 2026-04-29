#!/bin/bash
#SBATCH --job-name=exp2_4heads
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/exp2_%j.log

module load python/3.11.11-5e66 cuda/12.9.0-cinr
source ~/piano_env/bin/activate
export OSCAR_SCRATCH=/oscar/scratch/$USER/piano_transcription
python train.py --num_heads 4
