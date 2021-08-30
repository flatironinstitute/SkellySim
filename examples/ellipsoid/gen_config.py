#!/usr/bin/env python3

import sys
import copy
import toml
import numpy as np
import skelly_sim.param_tools as param_tools
from skelly_sim.skelly_config import ConfigEllipsoidal, Body, Fiber, unpack, perturb_fiber

if len(sys.argv) != 2:
    print("Supply output config path (blahblah.toml) as sole argument")
    sys.exit()

np.random.seed(100)

n_bodies = 0
n_fibers = 2000
fiber_template = Fiber()
fiber_template.length = 1.0
fiber_template.bending_rigidity = 2.5E-3
fiber_template.parent_body = -1
fiber_template.force_scale = -0.05
fiber_template.minus_clamped = True
fiber_template.n_nodes = 64

config = ConfigEllipsoidal()

config.bodies = [Body() for i in range(n_bodies)]
config.fibers = [copy.copy(fiber_template) for i in range(n_fibers)]
config.periphery.n_nodes = 8000

config.params.dt_write = 0.1
config.params.dt_initial = 8E-3
config.params.dt_max = 8E-3
config.params.velocity_field_flag = True
config.params.periphery_interaction_flag = False
config.params.velocity_field.moving_volume = False


def ellipsoid(t,
              u,
              a=config.periphery.a / 1.04,
              b=config.periphery.b / 1.04,
              c=config.periphery.c / 1.04):
    return np.array(
        [a * np.sin(u) * np.cos(t), b * np.sin(u) * np.sin(t), c * np.cos(u)])


n_trial_fibers = 5 * n_fibers
x = param_tools.r_surface(n_trial_fibers, ellipsoid, *(0, 2 * np.pi), *(0, np.pi))[0]
ds_min = 0.1
perturbation_amplitude = 0.005
i_trial_fiber = 0
for i in range(n_fibers):
    if i_trial_fiber >= n_trial_fibers:
        print(
            "Unable to insert fibers. Add more fiber trials, or decrease fiber density on the surface."
        )
        sys.exit()

    fib: Fiber = config.fibers[i]

    while (True):
        x0 = x[:, i_trial_fiber]

        reject = False
        for j in range(0, i - 1):
            if np.linalg.norm(x0 - config.fibers[j].x[0:3]) < ds_min:
                i_trial_fiber += 1
                reject = True
                break
        if reject:
            continue

        fiber_positions = perturb_fiber(perturbation_amplitude, fib.length, x0,
                                        fib.n_nodes)
        fib.x = fiber_positions.ravel().tolist()
        i_trial_fiber += 1
        break

with open(sys.argv[1], 'w') as fh:
    toml.dump(unpack(config), fh)
