cmake_minimum_required(VERSION 3.10)

# Get the current git commit
execute_process(
  COMMAND git describe --abbrev=8 --dirty --always
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE SKELLYSIM_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(SKELLYSIM_VERSION "0.10.0")
set(SKELLYSIM_TRAJECTORY_VERSION "1")
message("SkellySim version: v${SKELLYSIM_VERSION} (${SKELLYSIM_COMMIT}) (Trajectory: ${SKELLYSIM_TRAJECTORY_VERSION})")

project(
  skelly_sim
  VERSION ${SKELLYSIM_VERSION}
  LANGUAGES CXX CUDA
)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
option(EIGEN_USE_BLAS "Use BLAS as the backend for various Eigen calls" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using intel, disable offloading
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qno-offload")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
endif()
add_compile_options(-DSCTL_QUAD_T=__float128)

if(EIGEN_USE_BLAS)
  add_compile_definitions("EIGEN_USE_BLAS")
  message("Using BLAS backend for Eigen")
endif()

add_compile_options("$<$<CONFIG:RELEASE>:-g>")

add_definitions("-DSKELLYSIM_VERSION=\"${SKELLYSIM_VERSION}\"")
add_definitions("-DSKELLYSIM_COMMIT=\"${SKELLYSIM_COMMIT}\"")
add_definitions("-DSKELLYSIM_TRAJECTORY_VERSION=${SKELLYSIM_TRAJECTORY_VERSION}")

# required compiler features
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Kokkos REQUIRED HINTS $ENV{TRILINOS_BASE}/lib/cmake)
find_package(Tpetra REQUIRED HINTS $ENV{TRILINOS_BASE}/lib/cmake)
find_package(Teuchos REQUIRED HINTS $ENV{TRILINOS_BASE}/lib/cmake)
find_package(Belos REQUIRED HINTS $ENV{TRILINOS_BASE}/lib/cmake)
find_package(pvfmm REQUIRED)
find_package(STKFMM REQUIRED)
find_package(Boost REQUIRED)

set(SKELLY_INCLUDES ${PROJECT_SOURCE_DIR}/include)
set(SKELLY_INCLUDES_SYSTEM
  ${PROJECT_SOURCE_DIR}/extern/spdlog/include
  ${PROJECT_SOURCE_DIR}/extern/trng4
  ${PROJECT_SOURCE_DIR}/extern/toml11
  ${PROJECT_SOURCE_DIR}/extern/msgpack-c/include
  ${PROJECT_SOURCE_DIR}/extern/eigen
  ${MPI_CXX_INCLUDE_DIRS}
  ${PVFMM_INCLUDE_DIR}/pvfmm
  ${PVFMM_DEP_INCLUDE_DIR}
  ${STKFMM_INCLUDE_DIRS}
  ${Kokkos_INCLUDE_DIRS}
  ${Tpetra_INCLUDE_DIRS}
  ${Teuchos_INCLUDE_DIRS}
  ${Belos_INCLUDE_DIRS}
)

file(GLOB LIB_SRC src/core/*.cpp src/core/*.cu)
add_library(skelly OBJECT
  ${LIB_SRC}
  ${SKELLY_INCLUDES}
)

# Additions to keep the RPATH set for things like libmetis
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(INSTALL_RPATH_USE_LINK_PATH TRUE)
target_include_directories(skelly PUBLIC ${SKELLY_INCLUDES})
target_include_directories(skelly SYSTEM PUBLIC ${SKELLY_INCLUDES_SYSTEM})
target_link_libraries(skelly PUBLIC
  ${STKFMM_LIBRARIES}
  ${PVFMM_LIB_DIR}/${PVFMM_STATIC_LIB}
  ${PVFMM_DEP_LIB}
  OpenMP::OpenMP_CXX
  z
  MPI::MPI_CXX
  trng4_static
  ${Kokkos_LIBRARIES}
  ${Tpetra_LIBRARIES}
  ${Teuchos_LIBRARIES}
  ${Belos_LIBRARIES}
)

add_subdirectory(extern/spdlog EXCLUDE_FROM_ALL)
add_subdirectory(extern/trng4 EXCLUDE_FROM_ALL)

add_executable(skelly_sim src/skelly_sim.cpp)
target_link_libraries(skelly_sim PRIVATE skelly)

configure_file(misc/module_template.in module)

install(
  TARGETS skelly_sim
  CONFIGURATIONS Release
  RUNTIME DESTINATION bin
)

enable_testing()
add_custom_target(test_all ALL)
add_subdirectory(tests/core)
