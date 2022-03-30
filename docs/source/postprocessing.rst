.. _post-processing:

Post-processing
===============

SkellySim has a built-in mode for doing post-processing. Currently this is only used to
generate velocity fields, but could in the future be extended for other types of analysis.

:code:`$ mpirun skelly_sim --config-file=skelly_config.toml --post-process` will generate a
:code:`skelly_sim.vf` file that contains the velocity field, if your config file has the relevant
:code:`[params.velocity_field]` heading (see: :ref:`generating-configuration-files`). This can be visualized
with the provided visualization utilities in :ref:`visualization`.


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
