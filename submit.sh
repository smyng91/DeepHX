#!/bin/bash
#SBATCH -J DeepHX
#SBATCH --time=12:00:00
#SBATCH -A ldrd
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=a100:4
#SBATCH --mail-type="END"

module load horovod

python HX.py


