#!/usr/bin/env python3

import sys
import numpy as np
from skelly_sim.skelly_config import ConfigEllipsoidal, Body, Fiber

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

n_fibers = 2000

# create a config object and set the system parameters
config = ConfigEllipsoidal()
config.params.dt_write = 0.1
config.params.dt_initial = 8E-3
config.params.dt_max = 8E-3

# generate a list of fibers. They don't have positions though, so we need to fill those in
config.fibers = [
    Fiber(length=1.0,
          bending_rigidity=2.5E-3,
          parent_body=-1,
          force_scale=-0.05,
          minus_clamped=True,
          n_nodes=64) for i in range(n_fibers)
]

config.periphery.n_nodes = 8000

# move our fibers to the surface of the periphery and populate their position fields
config.periphery.move_fibers_to_surface(config.fibers, ds_min=0.1)

# output our config
config.save(config_file)

# uncomment the following to plot fibers on your ellipsoid
# config.plot_fibers()
