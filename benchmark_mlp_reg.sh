#!/bin/bash -l

#SBATCH -J benchmarkmlpreg 
#SBATCH -p gpu                
#SBATCH -t 40:00:00             
#SBATCH -N 1   
#SBATCH --gpus=1
#SBATCH --mem=50G                 
#SBATCH --output=benchmarkmlpreg.out

module load python
module load gcc
module load cuda
source ../../envs/mlproj/bin/activate

srun python benchmark_mlp_reg.py
