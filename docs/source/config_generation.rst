.. _generating-configuration-files:

Generating configuration files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single configuration file in the :obj:`toml` format must be provided to :obj:`SkellySim` in
order to run a simulation. This configuration file specifies input parameters as well as object
types and positions. While it is possible to write :obj:`SkellySim` configuration files by
hand, it's generally much more reliable and easier to generate the configurations from a python
script.

The python portion of :obj:`SkellySim` includes a utility module called :obj:`skelly_config`
to help generate these configuration files with some common configurations. It's advisable to
take a look at the `examples
<https://github.com/flatironinstitute/SkellySim/tree/main/examples>`_ to get you started on
complex configurations, but the basic outline is provided here, as well as a definition of all
parameters and some common terms.

Terminology
-----------

- Physical objects

  - :obj:`Fiber`: Flexible filament. Can be bound at either end to surfaces (:obj:`Bodies` and :obj:`Peripheries`),
    experience dynamic instability, and have constant motor forces per unit length along its axis.

  - :obj:`Body`: Mobile object (currently only a sphere) that can host fixed fibers or nucleate them via
    the dynamic instability module. Think: centrosomes, vesicles, and other objects that microtubules are
    typically associated with. Currently only spherical :obj:`Bodies` are supported.

  - :obj:`Periphery`: Fixed surface that :obj:`Fibers` and :obj:`Bodies` are contained within. Basically a fixed
    cellular wall. Spheres, Ellipses, and any convex surface of revolution are supported.

- Boundary Conditions

  - :obj:`clamped`: Boundary condition for :obj:`Fibers` where the relative orientation of the
    :obj:`Fiber` to its clamping point is preserved. For :obj:`Peripheries`, this means :obj:`(Velocity, AngularVelocity) = (0, 0)`
    and for :obj:`Bodies`, :obj:`(Velocity, AngularVelocity) = (VelocityBody, AngularVelocityBody)`

  - :obj:`hinged`: Boundary condition for :obj:`Fibers` where the position, but not the orientation, of the
    :obj:`Fiber` to its hinge point is preserved. For :obj:`Peripheries`, this means :obj:`(Velocity, Torque) = (0, 0)`
    and is not implemented for :obj:`Bodies`. See :obj:`Params.periphery_binding_flag`

- Configuration sections

  - :obj:`params` (required): Global system parameters. Things like timestepper parameters, and inter-object interaction specified here

  - :obj:`dynamic_instability` (optional): Parameters related to Fiber dynamic instability

  - :obj:`velocity_field` (optional): Parameters related to measuring the velocity field

  - :obj:`fibers` (optional): List of :obj:`Fiber` objects

  - :obj:`bodies` (optional): List of :obj:`Body` objects

  - :obj:`periphery` (optional): Single :obj:`Periphery` object

Example generation script
-------------------------

The following example script creates an initial configuration with a single :obj:`Body` at the origin
with a single :obj:`Fiber` attached to its surface. This object pair is contained within a spherical
:obj:`Periphery`.

.. highlight:: python
.. code-block:: python

    import numpy as np
    from skelly_sim.skelly_config import ConfigSpherical, Body, Fiber

    np.random.seed(100)

    # Create a config object and set the system parameters. System parameters
    # are typically things that control interobject interaction and
    # timestepping/measurement
    config = ConfigSpherical()
    config.params.eta = 1.0
    config.params.dt_initial = 1E-1
    config.params.dt_min = 1E-4
    config.params.dt_max = 1E-1
    config.params.dt_write = 1E-1
    config.params.t_final = 10.0
    config.params.gmres_tol = 1E-10
    config.params.seed = 130319

    # Bodies are stored in a list. Here we put one Body object at the origin
    # with 50 available nucleation sites. Since we're only adding one fiber, we
    # really one need one nucleation site, so this is overkill.  To give the
    # simulation some motion, we place a constant force in the 'y' direction on
    # the Body.
    config.bodies = [
        Body(n_nucleation_sites=50,
             position=[0.0, 0.0, 0.0],
             shape='sphere',
             radius=0.5,
             n_nodes=400,
             external_force=[0.0, 0.5, 0.0])
    ]

    # Fibers are stored in a list. Since all fiber nodes positions need to be
    # initialized (n_nodes * 3 numbers), we don't provide the position here, but
    # instead use a helper utility to generate the positions
    config.fibers = [
        Fiber(length=3.0,
              bending_rigidity=2.5E-3,
              parent_body=0,
              force_scale=0.0,
              n_nodes=64)
    ]

    # Move our fiber on the surface of the Body and populate the position array.
    config.fibers[0].fill_node_positions(x0=np.array([0.0, 0.0, 0.5]),
                                         normal=np.array([0.0, 0.0, 1.0]))

    # number of nodes to represent our containing spherical periphery. larger
    # peripheries = more nodes.
    config.periphery.n_nodes = 1000

    # radius of periphery in microns
    config.periphery.radius = 4.25

    # velocity field parameters.
    config.params.velocity_field.resolution = 1.0
    config.params.velocity_field.dt_write_field = 0.5
    # track field only around bodies when True. We want the whole field inside
    # the periphery
    config.params.velocity_field.moving_volume = False

    # output our config
    config.save('skelly_config.toml')

    # uncomment the following to plot fibers on your sphere
    # config.plot_fibers()


.. _skelly-config:

Configuration Parameters and API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: skelly_sim.skelly_config
   :members:
