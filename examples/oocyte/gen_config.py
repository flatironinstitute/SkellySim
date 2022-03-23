#!/usr/bin/env python3

import sys
import copy
import toml
import numpy as np
from skelly_sim.skelly_config import ConfigRevolution, Body, Fiber, unpack
from skelly_sim.shape_gallery import Envelope

if len(sys.argv) != 2:
    print("Supply output config path (blahblah.toml) as sole argument")
    sys.exit()

# This is just a script that creates a toml file using some utility functions I've written in in skelly_config
numpy_seed = 100
n_bodies = 0
n_fibers = 3000
np.random.seed(numpy_seed)

# Create our base fiber object, where all other fiber objects will be derived
fiber_template = Fiber()
fiber_template.length = 1.0
fiber_template.bending_rigidity = 2.5E-3
fiber_template.parent_body = -1
fiber_template.force_scale = -0.05
fiber_template.minus_clamped = True
fiber_template.n_nodes = 32

# config is the object that will actually get turned into a toml file at the end
config = ConfigRevolution()

# Make some bodies and some fibers. No bodies here, so thats's just empty
config.bodies = [Body() for i in range(n_bodies)]
config.fibers = [copy.copy(fiber_template) for i in range(n_fibers)]
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

# system wide parameters
config.params.dt_write = 0.1
config.params.dt_initial = 1E-2
config.params.dt_max = 1E-2
config.params.velocity_field_flag = True
config.params.periphery_interaction_flag = False
config.params.velocity_field.moving_volume = False
config.params.seed = 350
config.params.viscosity = 1

ds_min = 0.1
config.periphery.move_fibers_to_surface(config.fibers, ds_min)

# Scatter plot fiber beginning and end points. Note axes are not scaled, so results may look
# 'squished' and not uniform
import matplotlib.pyplot as plt
x_fib = np.array([fib.x[0:3] for fib in config.fibers])
x_fib_2 = np.array([fib.x[-3:] for fib in config.fibers])
ax = plt.axes(projection='3d')
ax.scatter(x_fib[:,0], x_fib[:,1], x_fib[:,2], color='blue')
ax.scatter(x_fib_2[:,0], x_fib_2[:,1], x_fib_2[:,2], color='green')
plt.show()

# output our config
with open(sys.argv[1], 'w') as fh:
    toml.dump(unpack(config), fh)
