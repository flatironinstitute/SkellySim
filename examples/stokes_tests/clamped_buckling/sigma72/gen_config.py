#!/usr/bin/env python3

import sys
import numpy as np
from skelly_sim.skelly_config import Config, Fiber, perturbed_fiber_positions

config_file = 'skelly_config.toml'
if len(sys.argv) == 1:
    print("Using default toml file for output: 'skelly_config.toml'. "
          "Provide an alternative filename argument to this script to use that instead.")
elif len(sys.argv) != 2:
    print("Supply output config path (blahblah.toml) as sole argument")
    sys.exit()
else:
    config_file = sys.argv[1]

np.random.seed(100)

# create a config object and set the system parameters
config = Config()
config.params.eta = 1.0
config.params.dt_initial = 1E-1
config.params.dt_write = 1.0
config.params.t_final = 10000.0
config.params.gmres_tol = 1E-10
config.params.seed = 130319
config.params.adaptive_timestep_flag = False

sigma = 0.0225
length = 2.0
bending_rigidity = 0.0025
n_nodes = 64

print(sigma * length**3 / bending_rigidity)

config.fibers = [Fiber(
    force_scale=-sigma,
    length=length,
    n_nodes=n_nodes,
    bending_rigidity=bending_rigidity,
    minus_clamped=True,
)]
x = perturbed_fiber_positions(0.01, length, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), n_nodes, np.array([1.0, 0.0, 0.0]))
config.fibers[0].x = x.ravel().tolist()

# output our config
config.save(config_file)
