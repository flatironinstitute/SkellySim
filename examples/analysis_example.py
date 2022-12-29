import numpy as np
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from skelly_sim.reader import TrajectoryReader

traj = TrajectoryReader('skelly_config.toml')
body_pos = np.empty(shape=(len(traj), 3))  # COM body position in time
plus_pos = np.empty(shape=(len(traj), 3))  # fiber plus end in time
minus_pos = np.empty(shape=(len(traj), 3))  # fiber minus end in time

for i in range(len(traj)):
    traj.load_frame(i)
    body_pos[i, :] = traj['bodies'][0]['position_']
    minus_pos[i, :] = traj['fibers'][0]['x_'][0, :]
    plus_pos[i, :] = traj['fibers'][0]['x_'][-1, :]

print("system keys: " + str(list(traj.keys())))
print("fiber keys: " + str(list(traj['fibers'][0].keys())))
print("body keys: " + str(list(traj['bodies'][0].keys())))
print("shell keys: " + str(list(traj['shell'].keys())))
print("velocity field keys: " + str(list(vf.keys())))

ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2, projection='3d')

ax1.plot(traj.times, body_pos[:, 2], traj.times, plus_pos[:, 2], traj.times, minus_pos[:, 2])
ax2.quiver(x[:, 0], x[:, 1], x[:, 2], v[:, 0], v[:, 1], v[:, 2], length=1.0, normalize=False)

plt.show()
