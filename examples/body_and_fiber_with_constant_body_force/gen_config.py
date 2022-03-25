#!/usr/bin/env python3

import sys
import numpy as np
from skelly_sim.skelly_config import Config, Body, Fiber

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

n_fibers = 1

# create a config object and set the system parameters
config = Config()
config.params.eta = 1.0
config.params.dt_initial = 1E-1
config.params.dt_min = 1E-4
config.params.dt_max = 1E-1
config.params.dt_write = 1E-1
config.params.t_final = 10.0
config.params.gmres_tol = 1E-10
config.params.seed = 130319

config.bodies = [
    Body(n_nucleation_sites=50,
         position=[0.0, 0.0, 0.0],
         shape='sphere',
         radius=0.5,
         n_nodes=400,
         external_force=[0.0, 0.0, 0.5])
]

# generate a list of fibers. They don't have positions though, so we need to fill those in
config.fibers = [
    Fiber(length=3.0,
          bending_rigidity=2.5E-3,
          parent_body=0,
          force_scale=0.0,
          minus_clamped=True,
          n_nodes=64) for i in range(n_fibers)
]
config.fibers[0].fill_node_positions(x0=np.array([0.0, 0.0, 0.5]),
                                     normal=np.array([0.0, 0.0, 1.0]))

config.params.velocity_field.resolution = 1.0
config.params.velocity_field.dt_write_field = 0.5
config.params.velocity_field.moving_volume = True
config.params.velocity_field.moving_volume_radius = 30.0

# output our config
config.save(config_file)

# uncomment the following to plot fibers on your ellipsoid
# config.plot_fibers()
