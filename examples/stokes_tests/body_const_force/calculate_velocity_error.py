#!/usr/bin/env python3

import numpy as np
from skelly_sim.reader import TrajectoryReader

traj = TrajectoryReader('skelly_config.toml')

# need beginning/end positions to calculate average velocity
traj.load_frame(0)
z_initial = traj['bodies'][0]['position_'][2]
traj.load_frame(len(traj) - 1)
z_final = traj['bodies'][0]['position_'][2]

dt = traj.times[-1] - traj.times[0]

# We need the hydrodynamic radius, which is slightly different than the supplied 'attachment'
# radius. let's just grab it from the precompute data.
precompute_data = np.load(traj.config_data['bodies'][0]['precompute_file'])
radius = np.linalg.norm(precompute_data["node_positions_ref"][0])
eta = traj.config_data['params']['eta']
force = traj.config_data['bodies'][0]['external_force'][2]

v_theoretical = force / (6 * np.pi * eta * radius)
v_measured = (z_final - z_initial) / dt

print("Measured velocity:    {}".format(v_measured))
print("Theoretical velocity: {}".format(v_theoretical))
print("Error |1 - v/v0|:     {}".format(abs(1 - v_measured / v_theoretical)))
