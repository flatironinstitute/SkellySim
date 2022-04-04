#!/usr/bin/env python3

import numpy as np
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from skelly_sim.reader import TrajectoryReader

traj = TrajectoryReader('skelly_config.toml')
vf = TrajectoryReader('skelly_config.toml', velocity_field=True)
body_z = np.empty(shape=(len(traj)))  # COM body position in time

for i in range(len(traj)):
    traj.load_frame(i)
    body_z[i] = traj['bodies'][0]['position_'][2]

v = np.diff(body_z) / np.diff(traj.times)

plt.plot(body_z[1:], v)
plt.xlabel('Body z')
plt.ylabel('Body velocity')
plt.ylim([0, 0.16])
plt.show()
