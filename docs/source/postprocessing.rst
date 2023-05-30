.. _post-processing:


Post-processing
---------------

For post-processing that require knowledge of the velocity at arbitrary points in the fluid,
SkellySim supports an "interactive" mode where it runs inside a containing python process in
:code:`Listener` mode. This allows you to exploit the SkellySim machinery on demand to
calculate various quantities without intermediate storage. This is extremely useful to generate
your own velocity fields, stream lines, and vortex lines at any given simulation point. Please
see the following to get started: `examples/listener_mode/listener_example.py
<https://github.com/flatironinstitute/SkellySim/blob/main/examples/listener_mode/listener_example.py>`_.


Trajectory format
-----------------

The trajectory format is a single file that consists of consecutive 'frames' with data
serialized in the :code:`msgpack` format. This makes doing post-processing in your language of
choice reasonably straightforward, though most existing code is centered around :code:`python`,
since that's what the configuration language and visualization are implemented in. Writing
readers in other languages, especially those that have builtin :code:`map` and :code:`list`
types, shouldn't be too much of a problem though.

Each frame of data is simply a nested dictionary of key-value pairs, which we provide a
convenience wrapper class for in :obj:`Python`. There is some example usage of this class in
:ref:`getting-started`, and we suggest users work from there or directly from
`examples/analysis_example.py
<https://github.com/flatironinstitute/SkellySim/blob/main/examples/analysis_example.py>`_.


TrajectoryReader class
----------------------

.. autoclass:: skelly_sim.reader.TrajectoryReader
   :members:


Listener class
--------------

.. autoclass:: skelly_sim.reader.Listener
   :members:


Request class
-------------

.. autoclass:: skelly_sim.reader.Request
   :members:


StreamlinesRequest class
------------------------

.. autoclass:: skelly_sim.reader.StreamlinesRequest
   :members:


VelocityFieldRequest class
--------------------------

.. autoclass:: skelly_sim.reader.VelocityFieldRequest
   :members:
