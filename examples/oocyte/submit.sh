#!/bin/bash -e
#SBATCH --partition=scc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --job-name=skelly_sim
#SBATCH --output=skelly_sim.log
#SBATCH --constraint=rome

module purge
module use ~rblackwell/modules
module load slurm
module load gcc/7.4.0
module load python3/3.7.3
module load intel/mkl/2020-4
module load intel/compiler/2020-4 -f
module load openmpi4/4.0.5-intel
module load home/trilinos/13.0.0-intel
module load home/pvfmm/903054-intel-openmpi4
module load home/stkfmm/bbcb21a-intel-openmpi4
module load home/skelly_sim/latest-intel

export OMP_NUM_THREADS=$((${SLURM_CPUS_ON_NODE}/${SLURM_NTASKS_PER_NODE}))
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

mpirun --map-by socket:pe=$OMP_NUM_THREADS skelly_sim --config-file=ooctye.toml
