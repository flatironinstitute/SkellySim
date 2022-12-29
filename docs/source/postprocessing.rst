.. _post-processing:


Interactive mode
----------------

SkellySim also supports an "interactive" mode where it runs inside a containing python process
in :code:`listener` mode. This allows you to exploit the SkellySim machinery on demand to
calculate various quantities without needless intermediate storage. This is extremely useful to
generate your own velocity fields, stream lines, and vortex lines at any given simulation
point. Eventually this will be the dominant post-processing mode, as more facilities are
added. Please see the following to get started:
`examples/listener_mode/listener_example.py
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


Python TrajectoryReader class
-----------------------------

.. autoclass:: skelly_sim.reader.TrajectoryReader
   :members:
