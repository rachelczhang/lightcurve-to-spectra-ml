#!/bin/bash -l
  
#SBATCH -J runcnn 
#SBATCH -p gpu                
#SBATCH -t 03:00:00             
#SBATCH -N 1   
#SBATCH --gpus=1
#SBATCH --mem=50G                 
#SBATCH --output=runcnn.out

module load modules/2.2
module load python
module load gcc
module load cuda
module load cudnn
source ../../envs/mlproj/bin/activate

srun python run_cnn.py
