cmake_minimum_required(VERSION 3.10)
project(
  fiber_sim
  VERSION 0.0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS OFF)

set(MPI_CXX_SKIP_MPICXX
    true
    CACHE BOOL "The MPI-2 C++ bindings are disabled.")

# required compiler features
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
# library
# find_package(pvfmm REQUIRED)
# find_package(STKFMM REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(fiber_sim src/fiber_sim.cpp)
target_include_directories(fiber_sim PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(fiber_sim PRIVATE STKFMM_STATIC Eigen3::Eigen OpenMP::OpenMP_CXX MPI::MPI_CXX)
