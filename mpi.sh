#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --tasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --constraint=cpu
#SBATCH --output=%j.log


export OMP_NUM_THREADS=32

srun -n 4  --cpu-bind=cores /global/cfs/cdirs/m4293/Trillions/TrillionsTest/MyApp