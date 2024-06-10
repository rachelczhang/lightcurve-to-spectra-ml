#!/bin/bash -l

#SBATCH -J readtessdata         
#SBATCH -p gen                   
#SBATCH -t 50:00:00             
#SBATCH -c 1                      
#SBATCH -N 1                      
#SBATCH --mem=50G                 
#SBATCH --output=readtess.out

module load python
source ../../envs/mlproj/bin/activate

srun python read_tess_data.py 

