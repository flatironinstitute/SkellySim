![](docs/source/images/SkellySim_Logo_RGB_Full.png)

SkellySim is a simulation package for simulating cellular components such as flexible filaments, motor proteins, and arbitrary rigid bodies.
It's designed to be highly scalable, capable of both OpenMP and MPI style parallelism, while using the efficient STKFMM/PVFMM libraries for hydrodynamic resolution.

# Documentation
Flatiron users should read [Installation/Running at FI first](#installation-running-at-fi),
since users there should not have to install the `C++` portion themselves.

[SkellySim documentation](https://users.flatironinstitute.org/~rblackwell/py-skellysim)

# Installation/Running at FI
## Basic setup
There are (currently) two components to skelly_sim, the python portion, and the actual
binary. The python portion is mostly for visualization, as well as generating config files and
precompute data. The binary (C++) portion is for actually running the simulation.

To install the python portion (in your virtual environment, conda environment, or using the `pip3 --user` option). For a virtualenv
```bash
module load python
python3 -m venv /path/to/my/virtualenv
source /path/to/my/virtualenv/bin/activate
pip3 install git+https://github.com/flatironinstitute/SkellySim
```
or for a conda environment
```
conda create -n myenvname
conda activate myenvname
pip3 install git+https://github.com/flatironinstitute/SkellySim
```
Due to the complex dependencies of the C++ portion, until I finish packaging things, you can use my modules. 
```bash
module -q purge
# REMOVE python module from this if you are using conda!!!!
module use ~rblackwell/modules
module -q load gcc/11 openmpi python trilinos pvfmm/1.3.0 intel-oneapi-mkl cuda flexiblas skelly_sim
```

## Building from source at FI (developers or externs)


```bash
module -q purge
module -q load gcc/11 openmpi python cmake trilinos pvfmm stkfmm intel-oneapi-mkl cuda boost flexiblas

git clone https://github.com/flatironinstitute/SkellySim
cd SkellySim
git submodule update --init --recursive
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-march=broadwell -DCMAKE_CUDA_ARCHITECTURES="70;75;80;85" -DBLA_VENDOR=FlexiBLAS
make -j$((2*$(nproc)))
```
