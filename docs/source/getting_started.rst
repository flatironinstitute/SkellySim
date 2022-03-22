.. _getting-started:

Getting started
===============

While there are a lot of complicated components to SkellySim, the basic workflow is fairly
straightforward.

1. :ref:`Generate a configuration file<generating-configuration-files>`

   :code:`$ python gen_config.py skelly_config.toml`
2. Generate precompute data

   :code:`$ skelly_precompute skelly_config.toml`
3. Run the simulation. Recommended in an :code:`sbatch` script. See example submission script (FIXME ADD SCRIPT).

   :code:`$ mpirun skelly_sim --config-file=skelly_config.toml`
4. :ref:`Visualize<visualization>` or :ref:`post-process<post-processing>`
