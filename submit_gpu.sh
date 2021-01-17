#!/bin/sh
#SBATCH --job-name="pinn_gpu"
#SBATCH -p genacc_q 
#SBATCH -t 2-00:00:00   
#SBATCH -n 12 
#SBATCH --gres=gpu:4

python HX.py
