import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from skelly_sim.reader import TrajectoryReader, Listener, Request

traj = TrajectoryReader('skelly_config.toml')
shell_radius = traj.config_data['periphery']['radius']
body_radius = traj.config_data['bodies'][0]['radius']

body_pos = np.empty(shape=(len(traj), 3)) # COM body position in time
plus_pos = np.empty(shape=(len(traj), 3)) # fiber plus end in time
minus_pos = np.empty(shape=(len(traj), 3)) # fiber minus end in time

for i in range(len(traj)):
    traj.load_frame(i)
    body_pos[i, :] = traj['bodies'][0]['position_']
    minus_pos[i, :] = traj['fibers'][0]['x_'][0, :]
    plus_pos[i, :] = traj['fibers'][0]['x_'][-1, :]

print("system keys: " + str(list(traj.keys())))
print("fiber keys: " + str(list(traj['fibers'][0].keys())))
print("body keys: " + str(list(traj['bodies'][0].keys())))
print("shell keys: " + str(list(traj['shell'].keys())))

ax1 = plt.subplot(1, 2, 2)

ax1.plot(traj.times, body_pos[:, 2], traj.times, plus_pos[:,2], traj.times, minus_pos[:,2])

# Fire up SkellySim in "listener" mode
listener = Listener(binary='skelly_sim_release')

# All analysis requests are done via a "Request" object
req = Request()

# specify frame number to evaluate and evaluator (CPU, GPU, FMM)
req.frame_no = 11
req.evaluator = "GPU"

# Request velocity field
tmp = np.linspace(-shell_radius, shell_radius, 20)
xm, ym, zm = np.meshgrid(tmp, tmp, tmp)
xcube = np.array((xm.ravel(), ym.ravel(), zm.ravel())).T
# Filter out points outside periphery and inside body
relpoints = np.where((np.linalg.norm(xcube - body_pos[11,:], axis=1) > body_radius) &
                     (np.linalg.norm(xcube, axis=1) < shell_radius))
req.velocity_field.x = xcube[relpoints]

# Make our request to SkellySim! Might take a second...
res = listener.request(req)

x = req.velocity_field.x
v = res['velocity_field']

ax2 = plt.subplot(1, 2, 2, projection="3d")
ax2.quiver(x[:, 0], x[:, 1], x[:, 2], v[:, 0], v[:, 1], v[:, 2])

plt.show()
