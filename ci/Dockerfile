FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV MKLROOT=/opt/intel/oneapi/mkl/latest
ENV STKFMM_ROOT=/usr/local
ENV CXXFLAGS_="-I/usr/lib/x86_64-linux-gnu/openmpi/include -march=broadwell"
ENV pvfmm_DIR="/usr/local/share/pvfmm"

RUN apt -y update && \
    apt -y install curl gpg software-properties-common && \
    curl -s https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
    apt-add-repository multiverse && \
    apt-add-repository universe && \
    apt -y update && \
    apt -y install libboost-dev intel-oneapi-mkl-devel build-essential libopenmpi-dev \
                   cmake libeigen3-dev ninja-build git python3-pip && \
    apt -y autoclean && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

RUN mkdir build

RUN cd build && \
    git clone https://github.com/dmalhotra/pvfmm && \
    cd pvfmm && \
    git checkout v1.3.0 && \
    git submodule update --init && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="$CXXFLAGS_" -GNinja && \
    ninja && \
    ninja install

RUN cd build && \
    git clone https://github.com/wenyan4work/STKFMM && \
    cd STKFMM && \
    mkdir build && \
    cd build && \
    pvfmm_DIR=/usr/local/share/pvfmm cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_CXX_FLAGS="$CXXFLAGS_" -GNinja && \
    ninja && \
    ninja install

RUN cd build && \
    curl -L -o trilinos.tar.gz https://github.com/trilinos/Trilinos/archive/refs/tags/trilinos-release-13-2-0.tar.gz && \
    tar -xf trilinos.tar.gz && \
    cd Trilinos-trilinos-release-13-2-0 && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DTrilinos_VERBOSE_CONFIGURE:BOOL=OFF -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_CXX_STANDARD:STRING=14 \
        -DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF -DTrilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
        -DTrilinos_ENABLE_OpenMP=ON -DTrilinos_ENABLE_Kokkos:BOOL=ON -DTrilinos_ENABLE_Belos:BOOL=ON \
        -DTrilinos_ENABLE_Tpetra:BOOL=ON -DTrilinos_ENABLE_Teuchos:BOOL=ON -DBUILD_TESTING:BOOL=OFF \
        -DTrilinos_ENABLE_Fortran:BOOL=OFF -D TPL_ENABLE_MPI=ON \
        -DTPL_BLAS_LIBRARIES=/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so \
        -DTPL_LAPACK_LIBRARIES=/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so \
        -DCMAKE_CXX_FLAGS="$CXXFLAGS_" -GNinja && \
    ninja && \
    ninja install

RUN rm -rf build

RUN mkdir -p $pvfmm_DIR && \
    cd $pvfmm_DIR && \
    curl -o Precomp_stokes_PVel_m16.data https://users.flatironinstitute.org/~rblackwell/Precomp_stokes_PVel_m16.data && \
    curl -o Precomp_stokes_vel_m16.data https://users.flatironinstitute.org/~rblackwell/Precomp_stokes_vel_m16.data
