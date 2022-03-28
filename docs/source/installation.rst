.. _installation:

Installation
============

There are (currently) two components to skelly_sim, the python portion, and the actual
binary. The python portion is used generating config files and precompute data as well as
visualization. The binary (:obj:`C++`) portion is for actually running the simulation and
generating field data. These two components are completely separate. This allows you to analyze
and visualize simulation data locally without having to install the simulation program.

Singularity (contains python and binary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For non-technical or new users just trying it out, we recommend just using our provided
:obj:`singularity` container. For cluster environments, we do recommend compiling from source,
though the singularity container should be more than adequate for single-node jobs. If you need
multi-node simulations, please compile using your cluster's MPI resources. I've provided two
builds: one for AVX, and one for AVX2 instruction sets. The AVX512 binary rarely performs
better enough to justify the maintenance of it. If you don't know what these are, AVX is a
safer bet and should provide good performance. If you're on a really ancient processor and AVX
is still too modern, I can provide a generic build if you reach out via our github issue
page. We don't officially support M1 macs, though you can likely get it to run on them with
limited to no vector instruction support.


- `Latest version (AVX) <https://users.flatironinstitute.org/~rblackwell/skellysim_singularity/skelly_sim_avx_latest.sif>`_
- `Latest version (AVX2) <https://users.flatironinstitute.org/~rblackwell/skellysim_singularity/skelly_sim_avx2_latest.sif>`_


Python modules and scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Simulation binary
~~~~~~~~~~~~~~~~~

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
