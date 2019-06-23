#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --account=def-sdraper
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=1

pwd
cd projects/anytime
pwd
source niagara/setup_env.sh

echo 'executing srun'
srun python -u mnist.py
# srun -n 2 python -u mnist.py
echo 'done srun!'
