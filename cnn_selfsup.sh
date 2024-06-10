#!/bin/bash -l
  
#SBATCH -J cnnselfsup 
#SBATCH -p gpu                
#SBATCH -t 02:00:00             
#SBATCH -N 1   
#SBATCH --gpus=1
#SBATCH --mem=50G                 
#SBATCH --output=cnnselfsup.out

module load python
module load gcc
module load cuda
module load cudnn
source ../../envs/mlproj/bin/activate

srun python cnn_selfsup.py

