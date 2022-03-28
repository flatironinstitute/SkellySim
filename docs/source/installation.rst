.. _installation:

Installation
============

There are (currently) two components to :obj:`SkellySim`, the python portion, and the actual
binary. The python portion is used generating config files and precompute data as well as
visualization. The binary (:obj:`C++`) portion is for actually running the simulation and
generating field data. These two components are completely separate. This allows you to analyze
and visualize simulation data locally without having to install the simulation program.

Singularity (beginner recommended: contains everything)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For non-technical or new users just trying it out, we recommend just using our provided
:obj:`singularity` container. For cluster environments, we do recommend compiling from source
if possible, though the singularity container should be more than adequate for single-node
jobs. If you absolutely need multi-node simulations, please compile using your cluster's MPI
resources (feel free to file a `github issue
<https://github.com/flatironinstitute/skellysim/issues>`_ if you are having issues. I've
provided two builds: one for AVX, and one for AVX2 instruction sets. The AVX512 binary rarely
performs better enough to justify the maintenance of it. If you don't know what these are, AVX
is a safer bet and should provide good performance. If you're on a really ancient processor and
AVX is still too modern, I can provide a generic build if you reach out via our github issue
page. We don't officially support M1 macs (you could possibly run under Rosetta, though we
wouldn't recommend it) and have not tested this on Intel Macs, though it will likely work on
Intel Macs.

- `Latest version (AVX) <https://users.flatironinstitute.org/~rblackwell/skellysim_singularity/skelly_sim_avx_latest.sif>`_
- `Latest version (AVX2) <https://users.flatironinstitute.org/~rblackwell/skellysim_singularity/skelly_sim_avx2_latest.sif>`_

Running commands in :obj:`singularity` containers is straightforward. First you need to
actually install singularity `singularity <https://sylabs.io/singularity>`_. At :obj:`flatiron`
and many other computing centers, this is available via the :obj:`module` system already (i.e. :obj:`module load singularity`).

Then, any command you would typically run directly in the shell, you just prefix it with
:obj:`singularity exec /path/to/image.sif`. So the workflow would look more like...

.. code-block:: bash

    singularity exec /path/to/skellysim/image.sif python3 gen_config.py
    singularity exec /path/to/skellysim/image.sif skelly_precompute skelly_config.toml
    singularity exec /path/to/skellysim/image.sif mpirun skelly_sim
    singularity exec /path/to/skellysim/image.sif mpirun skelly_sim --post-process


Note that this only works if the path you're writing to is in your home directory somewhere
directly, which in some environments isn't advisable to work from. Other paths (such as
:obj:`ceph` paths at FI), might need to be bound explicitly by singularity. To bind the current
working directory for read/write access you might have to change the command to be more like
this.

.. code-block:: bash

    singularity exec -B $PWD /path/to/skellysim/image.sif python3 gen_config.py


Python modules and scripts (advanced usage)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

virtualenv
----------

To install the python portion (in your virtual environment, conda environment, or using the :obj:`pip3 --user` option). For a virtualenv

.. highlight:: bash
.. code-block:: bash

    module load python # if you're using modules
    python3 -m venv /path/to/my/virtualenv
    source /path/to/my/virtualenv/bin/activate
    pip3 install git+https://github.com/flatironinstitute/SkellySim

Conda
-----

.. highlight:: bash
.. code-block:: bash

    conda create -n myenvname
    conda activate myenvname
    pip3 install git+https://github.com/flatironinstitute/SkellySim


Simulation binary (advanced usage)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Due to the complicated dependencies and the performance differences depending on what machine
you compile them to, it is difficult to provide general purpose binaries. If you don't need any
of this and don't want to deal with it, please just use the singularity builds. To get optimal
performance, or use multi-node MPI, you must build :obj:`SkellySim` and its dependencies from
source.

Building from source
--------------------

Requirements:

- `Trilinos 13 <https://github.com/trilinos/Trilinos/releases>`_ (with Kokkos, Belos, Teuchos, and Tpetra)
- `PVFMM <https://github.com/dmalhotra/pvfmm/releases>`_
- `STKFMM <https://github.com/wenyan4work/STKFMM/releases>`_
- BLAS/LAPACK (OpenBLAS or MKL or your implementations of choice)
- FFTW (FFTW3 or MKL-fftw)
- cmake (>=3.10)
- modern gcc (>=7). Should work with intel but not worth hassle in my tests

Will add a more detailed explanation here later, but please consult the `singularity build
script <https://github.com/flatironinstitute/SkellySim/blob/main/scripts/skelly_sim.def>`_ for a
general outline for how to build :obj:`PVFMM + STKFMM + Trilinos + SkellySim`.
