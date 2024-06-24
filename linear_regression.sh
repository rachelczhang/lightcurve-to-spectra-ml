#!/bin/bash -l
  
#SBATCH -J linreg
#SBATCH -p gen
#SBATCH -t 01:00:00
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --mem=50G
#SBATCH --output=linreg.out

module load python
source ../../envs/mlproj/bin/activate

srun python linear_regression.py
