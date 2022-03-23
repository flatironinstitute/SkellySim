#!/usr/bin/env python3

import sys
import toml
import numpy as np
from skelly_sim.skelly_config import ConfigRevolution, Body, Fiber
from skelly_sim.shape_gallery import Envelope

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

# This is just a script that creates a toml file using some utility functions I've written in in skelly_config
numpy_seed = 100
n_fibers = 3000
np.random.seed(numpy_seed)

# config is the object that will actually get turned into a toml file at the end
config = ConfigRevolution()

# system wide parameters
config.params.dt_write = 0.1
config.params.dt_initial = 1E-2
config.params.dt_max = 1E-2
config.params.velocity_field_flag = True
config.params.periphery_interaction_flag = False
config.params.velocity_field.moving_volume = False
config.params.seed = 350
config.params.viscosity = 1

# Create a list of identical base fibers to work with
config.fibers = [
    Fiber(length=1.0,
          bending_rigidity=2.5E-3,
          force_scale=-0.05,
          minus_clamped=True,
          n_nodes=32) for i in range(n_fibers)
]

# envelope is our surface of revolution. My algorithm tries to get the target nodes requested,
# but will likely return an object with slightly more nodes. This is a required option
config.periphery.envelope.n_nodes_target = 6000
# lower/upper bound are required options. ideally your function should go to zero at the upper/lower bounds
config.periphery.envelope.lower_bound = -3.75
config.periphery.envelope.upper_bound = 3.75
# required option. this is the function you're revolving around the 'x' axis. 'x' needs to be
# the independent variable. Currently the function has to be a one-liner
config.periphery.envelope.height = "0.5 * T * ((1 + 2*x/length)**p1) * ((1 - 2*x/length)**p2) * length"
# All the parameters that go into our height function
config.periphery.envelope.T = 0.72
config.periphery.envelope.p1 = 0.4
config.periphery.envelope.p2 = 0.2
config.periphery.envelope.length = 7.5

# minimum separation between fibers minus ends allowed on surface. distance is simple euclidean distance
ds_min = 0.1
config.periphery.move_fibers_to_surface(config.fibers, ds_min)

# output our config
config.save(config_file)

# just uncomment this to show a quick visualization of fiber positions
# config.plot_fibers()
