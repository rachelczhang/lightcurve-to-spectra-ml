#!/bin/bash -l

#SBATCH -J curvefitparams 
#SBATCH -p gen
#SBATCH -c 1
#SBATCH -t 10:00:00             
#SBATCH -N 1   
#SBATCH --mem=50G                 
#SBATCH --output=curvefit.out

module load python
source ../../envs/mlproj/bin/activate

srun python curvefit_params.py
