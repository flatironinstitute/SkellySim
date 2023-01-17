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
shell_precompute_data = np.load(traj.config_data['periphery']['precompute_file'])

r_body = np.linalg.norm(body_precompute_data["node_positions_ref"][0])
r_shell = np.linalg.norm(shell_precompute_data["nodes"][0])

lamb = r_body / r_shell

f = traj.config_data['bodies'][0]['external_force'][-1]
eta = traj.config_data['params']['eta']
gamma_theory = 6 * np.pi * r_body * 4 * eta * (1-lamb**5) / (4-9*lamb + 10*lamb**3 - 9*lamb**5 + 4*lamb**6)
v_theory = f / gamma_theory
gamma_compute = f / v[0]

print(r_body, r_shell, lamb)
print(v[0], v_theory, np.abs(1.0 - v[0]/v_theory))
print(gamma_compute, gamma_theory, np.abs(1.0 - gamma_compute/gamma_theory))

def mu_bar(z, a=r_body, R=r_shell):
    return 1 - 9 * a * (R*R / (R*R - z*z)) / 4 / R

gamma_vs_z = 6 * np.pi * eta * r_body / np.array(list(map(mu_bar, body_z[1:])))

plt.plot(body_z[1:], v)
plt.plot(body_z[1:], f / gamma_vs_z)
plt.xlabel('Body z')
plt.ylabel('Body velocity')
plt.legend(['SkellySim', '1 - 9 * a (R^2 / (R^2 - r^2)) / 4 / R'])
plt.title(f'Velocity of body in periphery a={r_body}, R={r_shell}')
#plt.ylim([0, 1.5])
plt.show()
