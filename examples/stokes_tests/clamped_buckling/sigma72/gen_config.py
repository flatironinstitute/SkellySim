#!/usr/bin/env python3

import numpy as np
from skelly_sim.skelly_config import Config, Fiber, Point

np.random.seed(100)

# create a config object and set the system parameters
config = Config()
config.params.eta = 1.0
config.params.dt_initial = 0.02
config.params.dt_max = 0.02
config.params.dt_write = 0.1
config.params.t_final = 200.0
config.params.pair_evaluator = "CPU"

sigma = 72.

length = 1.0
bending_rigidity = 0.0025
force_scale = -sigma * bending_rigidity / length**3
n_nodes = 32

config.fibers = [
    Fiber(
        force_scale=force_scale,
        length=length,
        n_nodes=n_nodes,
        bending_rigidity=bending_rigidity,
        minus_clamped=True,
    )
]

# give a small kick to start from
config.point_sources = [
    Point(
        position=[0.0, 0.0, 10 * length],
        force=[10.0, 0.0, 0.0],
        time_to_live=1.0,
    )
]

# orient in z
config.fibers[0].x = np.linspace([0, 0, 0], [0, 0, length], n_nodes).ravel().tolist()

# output our config
config.save()
