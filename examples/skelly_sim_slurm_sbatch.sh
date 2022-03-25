#!/bin/bash -e
#SBATCH --partition=scc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --job-name=skelly_sim
#SBATCH --output=skelly_sim.log
#SBATCH --constraint=rome

module purge
module use ~rblackwell/modules
module load gcc/11 openmpi intel-oneapi-mkl trilinos pvfmm stkfmm fftw skelly_sim
module list
ldd $(which skelly_sim)

export OMP_NUM_THREADS=$((${SLURM_CPUS_ON_NODE}/${SLURM_NTASKS_PER_NODE}))
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

srun --cpu-bind=ldoms skelly_sim --config-file=skelly_config.toml
