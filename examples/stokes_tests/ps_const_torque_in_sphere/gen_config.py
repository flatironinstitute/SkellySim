#!/usr/bin/env python3

import sys
import numpy as np
from skelly_sim.skelly_config import ConfigSpherical, Config, Point, Body

config_file = 'skelly_config.toml'
if len(sys.argv) == 1:
    print(
        "Using default toml file for output: 'skelly_config.toml'. "
        "Provide an alternative filename argument to this script to use that instead."
    )
elif len(sys.argv) != 2:
    print("Supply output config path (blahblah.toml) as sole argument")
    sys.exit()
else:
    config_file = sys.argv[1]

np.random.seed(100)

# create a config object and set the system parameters
# config = ConfigSpherical()
config = ConfigSpherical()
config.params.eta = 0.7
config.params.dt_initial = 1E-1
config.params.dt_min = 1E-4
config.params.dt_max = 1E-1
config.params.dt_write = 1E-1
config.params.t_final = 0.2
config.params.gmres_tol = 1E-10
config.params.seed = 130319

config.periphery.n_nodes = 4000
config.periphery.radius = np.pi / 3 / 1.04

config.point_sources = [
    Point(position=((np.random.uniform(size=3) - 0.5)*0.6).tolist(),
          torque=(np.random.uniform(size=3) - 0.5).tolist(),
          )
]

# output our config
config.save(config_file)
