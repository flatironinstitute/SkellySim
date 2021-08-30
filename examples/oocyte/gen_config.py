#!/usr/bin/env python3

import sys
import copy
import toml
import numpy as np
from skelly_sim.skelly_config import ConfigRevolution, Body, Fiber, unpack
from skelly_sim.shape_gallery import Envelope
from scipy.optimize import bisect

if len(sys.argv) != 2:
    print("Supply output config path (blahblah.toml) as sole argument")
    sys.exit()

# This is just a script that creates a toml file using some utility functions I've written in in skelly_config
numpy_seed = 100
n_bodies = 0
n_fibers = 3000
np.random.seed(numpy_seed)

# Create our base fiber object, where all other fiber objects will be derived
fiber_template = Fiber()
fiber_template.length = 1.0
fiber_template.bending_rigidity = 2.5E-3
fiber_template.parent_body = -1
fiber_template.force_scale = -0.05
fiber_template.minus_clamped = True
fiber_template.n_nodes = 32

# config is the object that will actually get turned into a toml file at the end
config = ConfigRevolution()

# Make some bodies and some fibers. No bodies here, so thats's just empty
config.bodies = [Body() for i in range(n_bodies)]
config.fibers = [copy.copy(fiber_template) for i in range(n_fibers)]
# envelope is our surface of revolution. My algorithm tries to get the target nodes requested,
# but will likely return an object with slightly more nodes. This is a required option
config.periphery.envelope.n_nodes_target = 6000
# lower/upper bound are required options. ideally your function should go to zero at the upper/lower bounds
config.periphery.envelope.lower_bound = -3.75
config.periphery.envelope.upper_bound = 3.75
# required option. this is the function you're revolving around the 'x' axis. 'x' needs to be
# the independent variable. Currently the function has to be a one-liner
config.periphery.envelope.height = "0.5 * T * ((1 + 2*x/length)**p1) * ((1 - 2*x/length)**p2) * length"
# All the parameters that go into our height function
config.periphery.envelope.T = 0.72
config.periphery.envelope.p1 = 0.4
config.periphery.envelope.p2 = 0.2
config.periphery.envelope.length = 7.5

# system wide parameters
config.params.dt_write = 0.1
config.params.dt_initial = 1E-2
config.params.dt_max = 1E-2
config.params.velocity_field_flag = True
config.params.periphery_interaction_flag = False
config.params.velocity_field.moving_volume = False
config.params.seed = 350
config.params.viscosity = 1


print("Constructing envelope function and its derivatives")
envelope = Envelope(config.periphery.envelope)
print("Envelope construction complete")

print("Inserting fibers")

# to generate microtubules on the surface, I build up a 'cdf' of ds(x)*h(x) (the infinitesimal
#  area of the shell at 'x'). Then solving 'cdf(x) = u' for x will give me a properly
#  distributed 'x'.
def build_cdf(f, lb, ub):
    # Evaluate function finely so that we can get a good estimate for the arc length
    xs = np.hstack([np.linspace(lb, ub, 1000000), [ub]])
    rs = f(xs)
    xd = np.diff(xs)
    rd = np.diff(rs)
    # shell area (off by some constant factor. using mean of height since diff() shortens
    # vectors)
    dist = np.sqrt(xd**2+rd**2) * (rs[0:-1] + rs[1:])
    # total arc length as function of x
    u = np.hstack([[0.0], np.cumsum(dist)]) / np.sum(dist)
    return xs, u

# invert a cdf to get a point uniformly distributed along the surface
def invert_cdf(y, xs, u):
    def f(x):
        return y - np.interp(x, xs, u)
    return bisect(f, xs[0], xs[-1])


xs, u = build_cdf(envelope.raw_height_func,
                  config.periphery.envelope.lower_bound,
                  config.periphery.envelope.upper_bound)
ds_min = 0.1
for i in range(n_fibers):
    fib: Fiber = config.fibers[i]
    print("Inserting fiber %d of %d" % (i + 1, n_fibers))
    i_trial = 0
    reject = True
    while (reject):
        i_trial += 1

        # generate trial 'x'
        x_trial = invert_cdf(np.random.uniform(), xs, u)
        h_trial = envelope.raw_height_func(x_trial)

        # generate trial 'y, z'
        theta = 2 * np.pi * np.random.uniform()
        y_trial = h_trial * np.cos(theta)
        z_trial = h_trial * np.sin(theta)

        # base of Fiber
        x0 = np.array([x_trial, y_trial, z_trial])

        # check for collisions
        reject = False
        for j in range(0, i - 1):
            if np.linalg.norm(x0 - config.fibers[j].x[0:3]) < ds_min:
                reject = True
                break
        if reject:
            continue

        # we need a normal, which requires derivatives. Our envelope can calculate them
        # arbitrarily.  However, the envelope object is a fit, and it doesn't necessarily fit
        # down to the supplied upper/lower bounds, if that function has a divergent
        # derivative. Here we just assume that unfit points points are aligned along 'x'
        if x0[0] < envelope.a:
            normal = np.array([1.0, 0.0, 0.0])
        elif x0[0] > envelope.b:
            normal = np.array([-1.0, 0.0, 0.0])
        else:
            # Use our envelope function to calculate the gradient/normal
            normal = np.array([envelope(x0[0]) * envelope.differentiate(x0[0]), -x0[1], -x0[2]])
            normal = normal / np.linalg.norm(normal)

        # Add fib.n_nodes points linearly along normal from the base
        fiber_positions = x0 + fib.length * np.linspace(0, normal, fib.n_nodes)

        # Assign these positions to fib.x
        fib.x = fiber_positions.ravel().tolist()

# Scatter plot fiber beginning and end points. Note axes are not scaled, so results may look
# 'squished' and not uniform
import matplotlib.pyplot as plt
x_fib = np.array([fib.x[0:3] for fib in config.fibers])
x_fib_2 = np.array([fib.x[-3:] for fib in config.fibers])
ax = plt.axes(projection='3d')
ax.scatter(x_fib[:,0], x_fib[:,1], x_fib[:,2], color='blue')
ax.scatter(x_fib_2[:,0], x_fib_2[:,1], x_fib_2[:,2], color='green')
plt.show()

# output our config
with open(sys.argv[1], 'w') as fh:
    toml.dump(unpack(config), fh)
