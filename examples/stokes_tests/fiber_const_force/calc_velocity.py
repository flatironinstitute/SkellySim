#!/usr/bin/env python3

from skelly_sim.reader import TrajectoryReader
import numpy as np

traj = TrajectoryReader('skelly_config.toml')

traj.load_frame(0)
x0 = traj['fibers'][0]['x_'][0, :]
traj.load_frame(-1)
xf = traj['fibers'][0]['x_'][0, :]

dt = traj.times[-1] - traj.times[0]
v = (xf - x0) / dt
fib = traj.config_data['fibers'][0]
epsilon = 1E-3

gamma = fib["force_scale"] * fib["length"] / v[-1]
gamma_theory = -4 * np.pi * fib['length'] * traj.config_data['params']['eta'] / (np.log(np.exp(1) * epsilon**2))

print("theoretical drag: {}".format(gamma_theory))
print("measured drag: {}".format(gamma))
print("relative error: {}".format(abs(1 - gamma/gamma_theory)))

