#!/bin/bash
#
#SBATCH --job-name=acgan
#SBATCH --cpus-per-task=4
#SBATCH --gpus a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --output=/scratch/jkbrook/Deep_Learning/HW4/log/acgan_ex.%j.out
#SBATCH --error=/scratch/jkbrook/Deep_Learning/HW4/log/acgan_ex.%j.ex

module load anaconda3 
source activate pytorch 

python3 acgan.py