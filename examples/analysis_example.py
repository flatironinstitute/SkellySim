import numpy as np
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from skelly_sim.reader import TrajectoryReader

traj = TrajectoryReader('skelly_config.toml')
vf = TrajectoryReader('skelly_config.toml', velocity_field=True)
body_pos = np.empty(shape=(len(traj), 3))

for i in range(len(traj)):
    traj.load_frame(i)
    body_pos[i, :] = traj['bodies'][0]['position_']

vf.load_frame(10)
x = vf['x_grid']
v = vf['v_grid']

ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2, projection='3d')

ax1.plot(traj.times, body_pos[:, 2])
ax2.quiver(x[:, 0], x[:, 1], x[:, 2], v[:, 0], v[:, 1], v[:, 2], length=1.0, normalize=False)

plt.show()
