#!/bin/sh
#SBATCH --job-name="pinn_gpu"
#SBATCH -p backfill 
#SBATCH -t 04:00:00   
#SBATCH -n 12 
#SBATCH --gres=gpu:4

python HX.py
