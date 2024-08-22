#!/bin/bash -l
  
#SBATCH -J regression 
#SBATCH -p gpu                
#SBATCH -t 01:00:00             
#SBATCH -N 1   
#SBATCH -C v100
#SBATCH --gpus=1
#SBATCH --mem=50G                 
#SBATCH --output=regression.out

module load modules/2.2
module load python
module load gcc
module load cuda
module load cudnn
source ../../envs/mlproj/bin/activate

srun python regression.py
