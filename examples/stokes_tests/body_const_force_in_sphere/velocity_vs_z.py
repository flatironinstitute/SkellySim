#!/usr/bin/env python3

import numpy as np
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from skelly_sim.reader import TrajectoryReader

traj = TrajectoryReader('skelly_config.toml')
body_z = np.empty(shape=(len(traj)))  # COM body position in time

for i in range(len(traj)):
    traj.load_frame(i)
    body_z[i] = traj['bodies'][0]['position_'][2]

v = np.diff(body_z) / np.diff(traj.times)

# Compare to theoretical velocity of confined sphere
body_precompute_data = np.load(traj.config_data['bodies'][0]['precompute_file'])
rs = np.linalg.norm(body_precompute_data["node_positions_ref"][0])
shell_precompute_data = np.load(traj.config_data['periphery']['precompute_file'])
rc = np.linalg.norm(shell_precompute_data["nodes"][0])
lamb = rs / rc

f = traj.config_data['bodies'][0]['external_force'][-1]
gamma_theory = 6 * np.pi * rs * 4 * (1-lamb**5) / (4-9*lamb + 10*lamb**3 - 9*lamb**5 + 4*lamb**6)
gamma_compute = f / v[0]

print(gamma_compute, gamma_theory, 1.0 - gamma_compute/gamma_theory)


plt.plot(body_z[1:], v)
plt.xlabel('Body z')
plt.ylabel('Body velocity')
plt.ylim([0, 0.16])
plt.show()
