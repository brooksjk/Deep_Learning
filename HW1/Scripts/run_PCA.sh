#!/bin/bash
#
#SBATCH --job-name=run_PCA
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --output=/scratch/jkbrook/Deep_Learning/HW_1/log/run_PCA_ex.%j.out
#SBATCH --error=/scratch/jkbrook//Deep_Learning/HW_1/log/run_PCA_ex.%j.ex


module load anaconda3
source activate pytorch

srun python3 HW1_1-2_PCA.py
