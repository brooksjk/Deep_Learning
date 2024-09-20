#!/bin/bash
#
#SBATCH --job-name=run_NumParams
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --output=/scratch/jkbrook/Deep_Learning/HW_1/log/run_NumParams_ex.%j.out
#SBATCH --error=/scratch/jkbrook//Deep_Learning/HW_1/log/run_NumParams_ex.%j.ex


module load anaconda3
source activate pytorch

cd ..

srun python3 HW1_1-3_NumParams.py
