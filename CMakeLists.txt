cmake_minimum_required(VERSION 3.10)
project(
  skelly_sim
  VERSION 0.0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS OFF)

# compiler flags
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using intel, disable offloading
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qno-offload")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qno-offload")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
endif()


set(MPI_CXX_SKIP_MPICXX
    true
    CACHE BOOL "The MPI-2 C++ bindings are disabled.")

# required compiler features
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pvfmm REQUIRED)
find_package(Kokkos REQUIRED HINTS $ENV{TRILINOS_BASE}/lib/cmake)
find_package(Tpetra REQUIRED HINTS $ENV{TRILINOS_BASE}/lib/cmake)
find_package(Teuchos REQUIRED HINTS $ENV{TRILINOS_BASE}/lib/cmake)
# find_package(Belos REQUIRED HINTS $ENV{TRILINOS_BASE}/lib/cmake)
# find_package(Ifpack REQUIRED HINTS $ENV{TRILINOS_BASE}/lib/cmake)

add_library(skelly STATIC src/fiber.cpp src/kernels.cpp)
target_include_directories(skelly PRIVATE ${PROJECT_SOURCE_DIR}/include ${PVFMM_INCLUDE_DIR}/pvfmm ${PVFMM_DEP_INCLUDE_DIR})
target_link_libraries(skelly PRIVATE  Eigen3::Eigen OpenMP::OpenMP_CXX MPI::MPI_CXX
  ${Kokkos_LIBRARIES} ${Tpetra_LIBRARIES} ${Teuchos_LIBRARIES}
  libSTKFMM_STATIC.a ${PVFMM_LIB_DIR}/${PVFMM_STATIC_LIB} ${PVFMM_DEP_LIB})

add_executable(skelly_sim src/skelly_sim.cpp)
target_include_directories(skelly_sim PRIVATE ${PROJECT_SOURCE_DIR}/include ${PVFMM_INCLUDE_DIR}/pvfmm ${PVFMM_DEP_INCLUDE_DIR})
target_link_libraries(skelly_sim PRIVATE skelly Eigen3::Eigen OpenMP::OpenMP_CXX MPI::MPI_CXX
  ${Kokkos_LIBRARIES} ${Tpetra_LIBRARIES} ${Teuchos_LIBRARIES}
  libSTKFMM_STATIC.a ${PVFMM_LIB_DIR}/${PVFMM_STATIC_LIB} ${PVFMM_DEP_LIB})

include(CTest)
add_subdirectory(tests)
