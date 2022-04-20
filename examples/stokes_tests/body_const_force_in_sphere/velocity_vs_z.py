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
eta = traj.config_data['params']['eta']
gamma_theory = 6 * np.pi * rs * 4 * eta * (1-lamb**5) / (4-9*lamb + 10*lamb**3 - 9*lamb**5 + 4*lamb**6)
v_theory = f / gamma_theory
gamma_compute = f / v[0]

print(v[0], v_theory)
print(gamma_compute, gamma_theory, 1.0 - gamma_compute/gamma_theory)

def mu_bar(r, a=rs, R=rc):
    # 10.1063/1.3681368
    alpha = 1 - 9 * a * (R*R / (R*R - r*r)) / 4 / R
    # beta = 1 - 9 * a * (4 * R**4 - 3*R**2*r**2 + r**4) / (R**2 - r**2) / 16 / R**3
    return alpha

gamma_vs_z = 6 * np.pi * eta * rs / np.array(list(map(mu_bar, body_z[1:])))


plt.plot(body_z[1:], v)
plt.plot(body_z[1:], f / gamma_vs_z)
plt.xlabel('Body z')
plt.ylabel('Body velocity')
#plt.ylim([0, 1.5])
plt.show()
