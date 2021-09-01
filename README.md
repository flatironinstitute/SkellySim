# SkellySim
SkellySim is a simulation package for simulating cellular components such as flexible filaments, motor proteins, and arbitrary rigid bodies.
It's designed to be highly scalable, capable of both OpenMP and MPI style parallelism, while using the efficient STKFMM/PVFMM libraries for hydrodynamic resolution.

# Running at FI
## Basic setup
There are (currently) two components to skelly_sim, the python portion, and the actual
binary. The python portion is mostly for visualization, as well as generating config files and
precompute data. The binary (C++) portion is for actually running the simulation.

To install the python portion (in your virtual environment, conda environment, or using the `pip3 --user` option). For a virtualenv
```bash
module load gcc python3
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
The best modules are likely to change, but this is good for now.
```bash
module purge;
module use ~rblackwell/modules
module load gcc/7.4.0 python3/3.7.3 intel/mkl/2020-4 intel/compiler/2020-4 openmpi4/4.0.5-intel \
    home/trilinos/13.0.0-intel home/pvfmm/903054-intel-openmpi4 home/stkfmm/bbcb21a-intel-openmpi4 \
    home/skelly_sim/latest-intel -f
```

## Run workflow 
The workflow is relatively straight forward. First you make a 'toml' config, which I've
provided some examples of in the 'examples' directory. You can either write this by hand, or use a python script to emit a config. The
hand-writing is likely to become completely deprecated, since it adds to code complexity on the C++ side that is hard to manage. I suggest looking at 
'examples/ellipsoid' and 'examples/oocyte' for the current best way to generate a config.

Next you need to make precompute data for that config. Note that the precompute only needs to be
redone if the 'Periphery' or 'Body' objects change their node positions, i.e. you add more nodes, or change the shape parameters.

Finally, you can run the simulation. `skelly_sim --config-file=myconfigfile.toml`

The basic workflow then simply looks like
```bash
mkdir my_simulation_dir
# make your gen_config.py
python3 gen_config.py myconfig.toml
skelly_sim --config-=file=myconfig.toml
```
though you should probably be using mpirun. See an example submit.sh script.
