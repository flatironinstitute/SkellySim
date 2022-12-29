#!/usr/bin/env python3

import numpy as np
from skelly_sim.reader import Listener, Request
import matplotlib as mpl
import matplotlib.pyplot as plt

# Fire up SkellySim in "listener" mode
listener = Listener(binary='skelly_sim_release')

# All analysis requests are done via a "Request" object
req = Request()

# specify frame number to evaluate and evaluator (CPU, GPU, FMM)
req.frame_no = 0
req.evaluator = "CPU"

# Request velocity field in z=0 plane
tmp = np.linspace(-2, 2, 1000)
xm, ym, zm = np.meshgrid(tmp, tmp, 0.0)
req.velocity_field.x = np.array((xm.ravel(), ym.ravel(), zm.ravel())).T

# Make our request to SkellySim! Might take a second...
res = listener.request(req)

# Streamplot of x,y components of velocity field in z=0 plane
ax1 = plt.subplot(1, 1, 1)
xm, ym = xm.squeeze(axis=2), ym.squeeze(axis=2)
U = res['velocity_field'][:, 0].reshape(*xm.shape)
V = res['velocity_field'][:, 1].reshape(*ym.shape)
ax1.streamplot(xm, ym, U, V)

# IT'S GO TIME
plt.show()
