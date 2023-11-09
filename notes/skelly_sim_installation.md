# SkellySim Instructions

## Installing the current version (11/7/2023 for body_ellipsoids branch)

This should be a set of instructions for compiling and installing the 'latest' version of SkellySim from source.

### Set up the modules
One needs to set up a consistent module environment for compilation.

```bash
module -q purge
module load modules/2.2-20230808
module load gcc/11.4.0
module load openmpi/4.0.7
module load trilinos/mpi-14.2.0
module load pvfmm
module load stkfmm
module load intel-oneapi-mkl/2023.1.0
module load cuda
module load boost
module load flexiblas
module load llvm
module load python/3.10.10
module load cmake
module list
```

It is sometimes convenient to put these into your `.bashrc` so that you can simply type `moduleskellysim`. For instance, my `.bashrc` has a section for SkellySim.

```bash
moduleskellysim() {
module -q purge
module load modules/2.2-20230808
module load gcc/11.4.0
module load openmpi/4.0.7
module load trilinos/mpi-14.2.0
module load pvfmm
module load stkfmm
module load intel-oneapi-mkl/2023.1.0
module load cuda
module load boost
module load flexiblas
module load llvm
module load python/3.10.10
module load cmake
}
```

### Getting the code and pre-installation
This gets the code from github, but also sets up the correct python virtual environment to use.

```bash
git clone git@github.com:flatironinstitute/SkellySim.git SkellySim
cd SkellySim/
git submodule update --init --recursive
git checkout body_ellipsoids
module load python
python3 -m venv --system-site-packages ~/envs/skellysim
source ~/envs/skellysim/bin/activate
pip install -e ./
```

### Compiling code

This basically follows the same instructions as are found online. Or use the module version found in Robert's
module system. First make sure you are in the directory for SkellySim. The following will install 
skelly\_sim to your local binary direction, in this case `~/bin/skelly_sim`.

```bash
cd <skelly_sim_directory>
<load modules as above>
module list
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-march=broadwell -DCMAKE_CUDA_ARCHITECTURES="70;75;86;90" -DBLA_VENDOR=FlexiBLAS -DCMAKE_INSTALL_PREFIX=~/
make -j$((2*$(nproc)))
make install
```

### Testing

Run the following tests (in the appropriate directories) to make sure everything is good before updating the branch. This will
run both the ctest version and the pytest versions.

```bash
cd build/
ctest
cd ../
PATH=$PWD/build:$PATH python3 -m pytest tests --verbose
```

### Next steps

You should have SkellySim now installed under `~/bin/skelly_sim`. You can then use this in slurm scripts, or just locally, to run things.

### Recompiling SkellySim after a source update

If SkellySim updates on github, you are going to need to get the changes and recompile and install it. In theory, the python bindings will always be set for your python environment. Otherwise, you want to execute the following steps.

```bash
git pull
cd build/
make -j16 install
```

This gets a new version of SkellySim and install it into wherever you have your install prefix set.


## Using SkellySim

### Using SkellySim interactively

First, you should make sure to run the `moduleskellysim` and `source ~/envs/skellysim/bin/activate` commands whenever you'd like to use SkellySim. A sample interactive experience would be something like the following.

```bash
ssh <cluster>
cd <path/to/simulations>
moduleskellysim
source ~/envs/skellysim/bin/activate
unset SLURM_JOBID
unset SLURM_JOB_ID
SPDLOG_LEVEL=info OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=threads mpirun -np 2 --map-by numa:pe=8 skelly_sim --overwrite
```

Notice that you need to unset the `SLURM_JOBID` and `SLURM_JOB_ID` environment variables if you are running an interactive session on the cluster.

### Generic slurm submission

Here is a generic file for slurm submission that you can use. The 3 lines at the end correspond to the configuration
step, the pre-compute step (which might be uneeded once you've done this the first time), and the simulation step itself.
You also might want to change the skelly\_generic.log for the job-name and output to something more useful for you.

`run_skellysim_generic.sh`

```bash
#!/bin/bash -e
#SBATCH --partition=ccb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --job-name=skelly_generic.log
#SBATCH --output=skelly_generic.log
#SBATCH --constraint=icelake
    
module -q purge
module load modules/2.2-20230808
module load gcc/11.4.0
module load openmpi/4.0.7
module load trilinos/mpi-14.2.0
module load pvfmm
module load stkfmm
module load intel-oneapi-mkl/2023.1.0
module load cuda
module load boost
module load flexiblas
module load llvm
module load python/3.10.10
module load cmake
module list
    
# Keep the exports for good measure
export OMP_NUM_THREADS=$((${SLURM_CPUS_ON_NODE}/${SLURM_NTASKS_PER_NODE}))
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_DISPLAY_ENV=true
    
# Source the version of the python stuff that goes with us
# For example
#source </path/to/your/skellysim/environment/bin/activate>
source ~/envs/skellysim/bin/activate
    
# Do whatever with the simulation
python3 config.py
# skelly_precompute skelly_config.toml
# mpirun --map-by socket:pe=$OMP_NUM_THREADS --report-bindings skelly_sim --config-file=skelly_config.toml --overwrite
```


## Analysis

### Creating a jupyterlab kernel (notebook, whatever)

If you want to login and use jupyterhub, you need to have a 'kernel' (or whatever) that knows where both the python and
executable portions of SkellySim live. In order to do this, we need to create a custom kernel, which can be done
following the instructions found on the flatiron wiki. Briefly, here is what to do to synchronize this kernel with 
the version you have been running.

```bash
Startup new shell on the cluster (rusty)
module -q purge
module load modules/2.2-20230808
module load gcc/11.4.0
module load openmpi/4.0.7
module load trilinos/mpi-14.2.0
module load pvfmm
module load stkfmm
module load intel-oneapi-mkl/2023.1.0
module load cuda
module load boost
module load flexiblas
module load llvm
module load python/3.10.10
module load cmake
source ~/envs/skellysim_foo/bin/activate
# source </path/to/your/skellysim/environment/bin/activate>
module load jupyter-kernels
python -m make-custom-kernel SkellySim_foo_jupyter
```

Now you have to add the custom ability to get into modules in this kernel, which is a pain. The way I found to do this 
is to create a wrapper script, and then modify it to load the proper modules.

```bash
# Change to the jupyter kernel directory you created
cd ~/.local/share/jupyter/kernels/skellysim_foo_jupyter
touch python-skellysimfoo-wrapper
chmod +x python-skellysimfoo-wrapper
cp kernel.json kernel.json.bak
```

Now go into the `python-skellysimfoo-wrapper` file and add the following. Make sure to set the source to the python environment
for this particular version of skellysim.

`python-skellysimdefoo-wrapper`

```bash
#!/bin/bash
source /mnt/sw/lmod/setup.sh
module -q purge
module load modules/2.2-20230808
module load gcc/11.4.0
module load openmpi/4.0.7
module load trilinos/mpi-14.2.0
module load pvfmm
module load stkfmm
module load intel-oneapi-mkl/2023.1.0
module load cuda
module load boost
module load flexiblas
module load llvm
module load python/3.10.10
module load cmake
source ~/envs/skellysim_foo/bin/activate
#source </path/to/your/skellysim/environment/bin/activate>
export PYTHONHOME=$PYTHON_BASE
exec python "$@"
```

Now edit `kernel.json` to call this python wrapper script, instead of the python environment. Here is an example kernel.json file, 
change the argv portion to match what you have!

`sample kernel.json`

```json
{
    "argv": ["/mnt/home/cedelmaier/.local/share/jupyter/kernels/skellysim_foo_jupyter/python-skellysimfoo-wrapper",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
    ],
    "display_name": "SkellySim_foo_jupyter",
    "language": "python",
    "metadata": {
    "debugger": true
    }
}
```

### Analysis (Command line)

In many cases it is better to use the command line to run the analysis, as I sometimes have issues with Jupyter working
correctly, especially when calling other scripts via the command line. You can basically use the same setup as the 
jupyter portion, and then just directly execute the python code. Here is an example configuration that I was testing
against a large spherical periphery with 1318 fibers. First, I would login to a Rome node (or something similar) as there
are some pretty big memory requirements depending on the size of your periphery. You're going to want to first startup
your entire environment (SkellySim executable + Python information) before trying to run this.

`analysis_test.py`

```python
#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from skelly_sim.reader import TrajectoryReader, Listener, Request

traj = TrajectoryReader('skelly_config.toml')
shell_radius = traj.config_data['periphery']['radius']

plus_pos = np.empty(shape=(len(traj), 3)) # fiber plus ends
minus_pos = np.empty(shape=(len(traj), 3)) # fiber minus ends

for i in range(len(traj)):
    traj.load_frame(i)
    minus_pos[i, :] = traj['fibers'][0]['x_'][0, :]
    plus_pos[i, :] = traj['fibers'][0]['x_'][-1, :]

print("system keys: " + str(list(traj.keys())))
print("fiber keys: " + str(list(traj['fibers'][0].keys())))
print("shell keys: " + str(list(traj['shell'].keys())))
print(f"trajectory length: {len(traj)}")

fig1, ax1 = plt.subplots(1, 1)
ax1.plot(traj.times, plus_pos[:,2])
ax1.plot(traj.times, minus_pos[:,2])
fig1.savefig('test_endpoint_positions.pdf', dpi = fig1.dpi)

# Fire up the skellysim listener object
#listener = Listener(binary='skelly_sim')
listener = Listener(binary='mpirun -np 1 skelly_sim')

# All analysis are done via the requester
req = Request()

# Specify a frame to get stuff
req.frame_no = 299
req.evaluator = "FMM"
#req.evaluator = "GPU"

# Request velocity field
tmp = np.linspace(-shell_radius, shell_radius, 15)
xm, ym, zm = np.meshgrid(tmp, tmp, tmp)
xcube = np.array((xm.ravel(), ym.ravel(), zm.ravel())).T

# Filter out the points outside the periphery
relpoints = np.where(np.linalg.norm(xcube, axis=1) < shell_radius)
req.velocity_field.x = xcube[relpoints]

# Make the request of the listener
res = listener.request(req)

x = req.velocity_field.x
print(f"x: {x}")
v = res['velocity_field']
print(f"v: {v}")

#fig2, ax2 = plt.subplots(1, 2, 2, projection="3d")
#fig2, ax2 = plt.figure().add_subplot(projection="3d")
fig2 = plt.figure()
ax2 = plt.axes(projection='3d')
ax2.quiver(x[:,0], x[:,1], x[:,2], v[:,0], v[:,1], v[:,2])
fig2.savefig('test_velocityfield.pdf', dpi = fig2.dpi)
```

### Analysis (Bodies)

Here is the information stored for the bodies in each frame. Assuming a body index `ibody` in the sequence of multiple bodies. This is just the information written to the trajectory at time `times = traj.times`.

```python
# Center of mass position [3x1 vector]
position: traj['bodies'][ibody]['position_']
# Orientation quanterion [4x1 quaternion]
orientation: traj['bodies'][ibody]['orientation_']
# Complete solution vector [3*n_nodes densities, 3x1 velocity, 3x1 angular velocity]
solution_vec: traj['bodies'][ibody]['solution_vec_']
velocity: traj['bodies'][ibody]['solution_vec_'][-6:-3]
angular_velocity: traj['bodies'][ibody]['solution_vec_'][-3:]
```


## Installing the previous version (def3f80c) (5/9/2023) (maybe deprecated 11/9/2023)

This is the last 'good' version of SkellySim that synchronizes with Robert's module version. These are the instructions
for installing a specific git has (in this case de3f80c) in order to test differences, including how the python
files are handled.

### Getting the code and pre-installation

    git clone git@github.com:flatironinstitute/SkellySim.git SkellySim_de3f80c
    cd SkellySim_de3f80c/
    git reset --hard de3f80c
    git submodule update --init --recursive
    module load python
    python3 -m venv --system-site-packages ~/envs/skellysim_de3f80c
    source ~/envs/skellysim_de3f80c/bin/activate
    pip install -e ./

### Compiling code

This basically follows the same instructions as are found online. Or use the module version found in Robert's
module system. First make sure you are in the directory for SkellySim. The following will install 
skelly\_sim to your local binary direction, in this case `~/bin/skelly_sim`.

```
cd <skelly_sim_directory>
module -q purge
module -q load slurm gcc/11 openmpi python cmake trilinos pvfmm stkfmm intel-oneapi-mkl cuda boost flexiblas
module list
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-march=broadwell -DCMAKE_CUDA_ARCHITECTURES="70;75;86;90" -DBLA_VENDOR=FlexiBLAS -DCMAKE_INSTALL_PREFIX=~/
make -j$((2*$(nproc)))
make install

```

### **(WORKAROUND)** Removing issue where fiber is interacting with periphery

This fix is to change when the fiber is interacting with the periphery on its (-) end.
**MAKE SURE YOU ARE EDITING THE CORRECT VERSION OF THE PYTHON FILE THAT GOES WITH YOUR
INSTALLATION OF SKELLYSIM!!!!!**

You can add the following lines to the end of your `config.py` file (or whatever you are naming it).

    import skelly_sim
    print(skelly_sim.__file__)

This will tell you the actual version of the SkellySim python you are using, just to make sure everything
is synchronized. You want to go edit the `skelly_config.py` file associated with this installation, and check
for the radius stand-off distance, which is originally set to 0.99999..) and change to something like 0.999.

### **(WORKAROUND)** Issues with mpirun and skelly\_sim during analysis

SLURM complains when running SkellySim in listener mode as invoked by python during analysis. To fix this, we can
change how Python opens SkellySim in listener mode. You're going to need to find the copy of `reader.py` that is loaded, and then make the following change.

    #self._proc = Popen([binary, '--listen'], stdin=PIPE, stdout=PIPE)
    self._proc = Popen((binary + ' --listen').split(), stdin=PIPE, stdout=PIPE)

Then, in your `config.py` file that actually runs the analysis, use the following type syntax to invoke skelly\_sim with mpirun.

    #listener = Listener(binary='skelly_sim')
    listener = Listener(binary='mpirun -np 1 skelly_sim')