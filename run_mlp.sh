#!/bin/bash -l

#SBATCH -J runmlp 
#SBATCH -p gpu                
#SBATCH -t 00:50:00             
#SBATCH -C v100
#SBATCH -N 1   
#SBATCH --gpus=1
#SBATCH --mem=50G                 
#SBATCH --output=runmlp.out

module load python
module load gcc
module load cuda
source ../../envs/mlproj/bin/activate

srun python run_mlp.py 

