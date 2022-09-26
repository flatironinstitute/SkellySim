#!/usr/bin/env python3

import numpy as np
from skelly_sim.reader import Listener, Request
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TKAgg')

# Fire up SkellySim in "listener" mode
listener = Listener()

# All analysis requests are done via a "Request" object
req = Request()

# specify frame number to evaluate and evaluator (CPU, GPU, FMM)
req.frame_no = 1
req.evaluator = "GPU"

# Request 3 streamlines from t=[-10, 10]
req.streamlines.x0 = np.array([
    [0.25, 0.0, 0.0],    
    [0.5, 0.0, 0.0],
    [1.0, 0.0, 0.0],
])
req.streamlines.t_final = 10.0

# Request single vortex line from t=[-10, 10]
req.vortexlines.x0 = np.array([
    [0.0, 0.0, 2.0],
])
req.vortexlines.t_final = 10.0

# Request velocity field in z=0 plane
tmp = np.linspace(-2, 2, 100)
xm, ym, zm = np.meshgrid(tmp, tmp, 0.0)
req.velocity_field.x = np.array((xm.ravel(), ym.ravel(), zm.ravel())).T

# Make our request to SkellySim! Might take a second...
res = listener.request(req)

# Plot our streamlines
ax1 = plt.subplot(1, 2, 1, projection='3d')
for sl in res['streamlines']:
    x = sl['x'] # Points along streamline
    v = sl['val'] # Vector velocities at streamline evaluation points. Not used here, but available
    t = sl['time'] # time of evals. Not used here, but available

    ax1.plot3D(x[:,0], x[:,1], x[:,2])


# Streamplot of x,y components of velocity field in z=0 plane
ax2 = plt.subplot(1, 2, 2)
xm, ym = xm.squeeze(axis=2), ym.squeeze(axis=2)
U = res['velocity_field'][:, 0].reshape(*xm.shape)
V = res['velocity_field'][:, 1].reshape(*ym.shape)
ax2.streamplot(xm, ym, U, V)

# IT'S GO TIME
plt.show()
