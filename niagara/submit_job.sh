#!/bin/bash
#SBATCH --time=00:16:00
#SBATCH --account=def-sdraper
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1

pwd
cd ~/projects/anytime
pwd
source niagara/setup_env.sh

echo 'executing srun'
srun python -u run_eval.py
# srun -n 2 python -u mnist.py
echo 'done srun!'
