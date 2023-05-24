#!/usr/bin/env python3

from skelly_sim.reader import TrajectoryReader

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

def get_deflection(path: str):
    traj = TrajectoryReader(path)

    x = []
    for i in range(len(traj)):
        traj.load_frame(i)
        x.append(traj['fibers'][0]['x_'][-1, 0])

    return traj.times, x

t72, x72 = get_deflection('sigma72/skelly_config.toml')
t80, x80 = get_deflection('sigma80/skelly_config.toml')

plt.plot(t72, x72, t80, x80)
plt.legend(['\u03c3 = 72', '\u03c3 = 80'])
plt.xlabel('Time (s)')
plt.ylabel('Deflection (micrometers)')
plt.show()
