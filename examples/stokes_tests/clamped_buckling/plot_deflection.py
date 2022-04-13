#!/usr/bin/env python3

from skelly_sim.reader import TrajectoryReader
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

traj = TrajectoryReader('skelly_config.toml')

x = []
for i in range(len(traj)):
    traj.load_frame(i)
    x.append(traj['fibers'][0]['x_'][-1, 0])

plt.plot(traj.times, x)
plt.xlabel('Time (s)')
plt.ylabel('Deflection (micrometers)')
plt.show()
