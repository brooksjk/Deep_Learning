#!/bin/bash
#
#SBATCH --job-name=run_GradNorm
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --output=/scratch/jkbrook/Deep_Learning/HW_1/log/run_GradNorm__ex.%j.out
#SBATCH --error=/scratch/jkbrook//Deep_Learning/HW_1/log/run_GradNorm__ex.%j.ex


module load anaconda3
source activate pytorch

cd ..

srun python3 HW1_1-2_GradNorm.py
