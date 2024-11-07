#!/bin/bash
#
#SBATCH --job-name=squad_qa
#SBATCH --cpus-per-task=4
#SBATCH --gpus a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=/scratch/jkbrook/Deep_Learning/HW_3/log/test_ex.%j.out
#SBATCH --error=/scratch/jkbrook//Deep_Learning/HW_3/log/test_ex.%j.ex

module load anaconda3 
source activate pytorch 

python3 finetuned_model.py