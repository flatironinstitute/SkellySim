#!/usr/bin/env python3

import sys
import numpy as np
from skelly_sim.skelly_config import Config, Body

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
config.params.eta = 0.9
config.params.dt_initial = 1E-1
config.params.dt_min = 1E-4
config.params.dt_max = 1E-1
config.params.dt_write = 1E-1
config.params.t_final = 10.0
config.params.gmres_tol = 1E-10
config.params.seed = 130319
config.params.pair_evaluator = "CPU"

config.bodies = [
    Body(n_nucleation_sites=0,
         position=[0.0, 0.0, 0.0],
         shape='sphere',
         radius=0.5,
         n_nodes=400,
         external_force=[0.0, 0.0, 1.5])
]

config.save(config_file)
