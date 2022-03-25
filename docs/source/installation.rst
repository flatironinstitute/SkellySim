.. _installation:

Installation
============

There are (currently) two components to skelly_sim, the python portion, and the actual
binary. The python portion is used generating config files and precompute data as well as
visualization. The binary (:obj:`C++`) portion is for actually running the simulation and
generating field data. These two components are completely separate. This allows you to analyze
and visualize simulation data locally without having to install the simulation program.

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

SkellySim has a number of requirements to work. A superbuild is in progress that only requires
the user have a working MPI installation, cmake, and gcc, fftw, and a blas implementation (all
things installable via conda), but that is very much a work in progress.

Requirements:

- `Trilinos 13 <https://github.com/trilinos/Trilinos/releases>`_ (with Kokkos, Belos, Teuchos, and Tpetra)
- `PVFMM <https://github.com/dmalhotra/pvfmm/releases>`_
- `STKFMM <https://github.com/wenyan4work/STKFMM/releases>`_
- BLAS/LAPACK (OpenBLAS or MKL or your implementations of choice)
- FFTW (FFTW3 or MKL-fftw)
- cmake (>=3.10)
- modern gcc (>=7)

