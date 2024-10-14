#!/bin/bash
#
#SBATCH --job-name=train
#SBATCH --cpus-per-task=4
#SBATCH --gpus a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=/scratch/jkbrook/Deep_Learning/HW_2/hw2_1/log/train_ex.%j.out
#SBATCH --error=/scratch/jkbrook//Deep_Learning/HW_2/hw2_1/log/train_ex.%j.ex


module load anaconda3
source activate pytorch

python3 training.py