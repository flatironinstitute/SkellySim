#!/usr/bin/env python3

from skelly_sim.reader import TrajectoryReader

traj = TrajectoryReader('skelly_config.toml')
vf = TrajectoryReader('skelly_config.toml', velocity_field=True)

traj.load_frame(0)
x0 = traj['fibers'][0]['x_'][0, :]
traj.load_frame(len(traj) - 1)
xf = traj['fibers'][0]['x_'][0, :]

dt = traj.times[-1] - traj.times[0]
v = (xf - x0) / dt
print(v[-1])
