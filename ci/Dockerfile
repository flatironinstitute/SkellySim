FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV MKLROOT=/opt/intel/oneapi/mkl/latest
ENV CXXFLAGS_="-I/usr/lib/x86_64-linux-gnu/openmpi/include -march=native"

RUN apt -y update && \
    apt -y install curl gpg software-properties-common && \
    curl -s https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
    apt-add-repository multiverse && \
    apt-add-repository universe && \
    apt -y update && \
    apt -y install intel-oneapi-mkl-devel build-essential libopenmpi-dev cmake libeigen3-dev ninja-build git python3-pip && \
    apt -y autoclean

RUN mkdir build

RUN cd build && \
    curl -L -o pvfmm.tar.gz https://github.com/dmalhotra/pvfmm/archive/refs/tags/v1.3.0.tar.gz && \
    tar -xf pvfmm.tar.gz && \
    mkdir pvfmm-build && \
    cd pvfmm-build && \
    cmake ../pvfmm-1.3.0 -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS="$CXXFLAGS_" -GNinja && \
    ninja && \
    ninja install && \

RUN cd build && \
    curl -L -o stkfmm.tar.gz https://github.com/wenyan4work/STKFMM/archive/refs/tags/v1.1.0.tar.gz && \
    tar -xf stkfmm.tar.gz && \
    mkdir stkfmm-build && \
    pushd stkfmm-build && \
    pvfmm_DIR=/usr/local/share/pvfmm cmake ../STKFMM-1.0.0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
         -DCMAKE_CXX_FLAGS="$CXXFLAGS_" -GNinja && \
    ninja && \
    ninja install && \

RUN cd build && \
    curl -L -o trilinos.tar.gz https://github.com/trilinos/Trilinos/archive/refs/tags/trilinos-release-13-2-0.tar.gz && \
    tar -xf trilinos.tar.gz && \
    mkdir trilinos-build && \
    pushd trilinos-build && \
    cmake ../Trilinos-trilinos-release-13-2-0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DTrilinos_VERBOSE_CONFIGURE:BOOL=OFF -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_CXX_STANDARD:STRING=14 -DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF -DTrilinos_ENABLE_ALL_PACKAGES:BOOL=OFF -DTrilinos_ENABLE_OpenMP=ON -DTrilinos_ENABLE_Kokkos:BOOL=ON -DTrilinos_ENABLE_Belos:BOOL=ON -DTrilinos_ENABLE_Tpetra:BOOL=ON -DTrilinos_ENABLE_Teuchos:BOOL=ON -DBUILD_TESTING:BOOL=OFF -DTrilinos_ENABLE_Fortran:BOOL=OFF -D TPL_ENABLE_MPI=ON -DTPL_BLAS_LIBRARIES=/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so -DTPL_LAPACK_LIBRARIES=/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so -DCMAKE_CXX_FLAGS="$CXXFLAGS_" -GNinja && \
    ninja && \
    ninja install && \

RUN rm -rf build
    # git clone https://github.com/flatironinstitute/skellysim
    # cd skellysim
    # git checkout ${SKELLY_VERS}
    # git submodule update --init
    # cd ..

    # mkdir skellysim-build
    # cd skellysim-build
    # STKFMM_ROOT=/usr/local cmake ../skellysim -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
    # -DCMAKE_CXX_FLAGS="$CXXFLAGS_" -DBLAS_LIBRARIES=$MKLROOT/lib/intel64/libmkl_rt.so \
    # -DLAPACK_LIBRARIES=$MKLROOT/lib/intel64/libmkl_rt.so -GNinja
    # ninja
    # ninja install

    # cd ../..
    # rm -rf build

# %labels
#     Author blackwer
