#!/usr/bin/env python3
import numpy as np

from skelly_sim.reader import TrajectoryReader

vf = TrajectoryReader('skelly_config.toml', velocity_field=True)

vf.load_frame(1)
x : np.array = vf['x_grid']
v : np.array = vf['v_grid']
