.. _generating-configuration-files:

Generating configuration files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While SkellySim has limited facilities for writing config files by hand, it's usually much
better to generate the configurations from a python script. The python portion of `skelly_sim`
includes a utility module called `skelly_config` to help generate configuration files. It's
advisable to take a look at the `examples
<https://github.com/flatironinstitute/SkellySim/tree/main/examples>`_ to get you started on
complex configurations, but the basic outline is provided here, as well as a definition of all
parameters.

.. highlight:: python
.. code-block:: python

   dostuff()

.. automodule:: skelly_sim.skelly_config
   :members:
