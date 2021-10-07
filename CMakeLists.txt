cmake_minimum_required(VERSION 3.10)

# Get the current git tag
execute_process(
  COMMAND git describe --tags
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_TAG
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "v" "" GIT_TAG ${GIT_TAG})

project(
  skelly_sim
  VERSION ${GIT_TAG}
  LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

option(EIGEN_USE_BLAS "Use BLAS as the backend for various Eigen calls" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare -Wno-unused-variable")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-value -Wno-misleading-indentation")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using intel, disable offloading
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qno-offload")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
endif()

if(EIGEN_USE_BLAS)
  add_compile_definitions("EIGEN_USE_BLAS")
  message("Using BLAS backend for Eigen")
endif()


# required compiler features
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pvfmm REQUIRED)
find_package(STKFMM REQUIRED)
find_package(Kokkos REQUIRED HINTS $ENV{TRILINOS_BASE}/lib/cmake)
find_package(Tpetra REQUIRED HINTS $ENV{TRILINOS_BASE}/lib/cmake)
find_package(Teuchos REQUIRED HINTS $ENV{TRILINOS_BASE}/lib/cmake)
find_package(Belos REQUIRED HINTS $ENV{TRILINOS_BASE}/lib/cmake)

add_library(skelly STATIC src/fiber.cpp src/kernels.cpp src/utils.cpp src/periphery.cpp src/cnpy.cpp src/params.cpp
  src/system.cpp src/body.cpp src/solver_hydro.cpp src/rng.cpp src/spherical_body.cpp src/deformable_body.cpp src/point_source.cpp)

set(SKELLY_INCLUDES
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/extern/spdlog/include
  ${PROJECT_SOURCE_DIR}/extern/trng4
  ${PROJECT_SOURCE_DIR}/extern/toml11
  ${PROJECT_SOURCE_DIR}/extern/msgpack-c/include
  ${MPI_CXX_INCLUDE_DIRS}
  ${PVFMM_INCLUDE_DIR}/pvfmm
  ${PVFMM_DEP_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${STKFMM_INCLUDE_DIRS}
  ${Kokkos_INCLUDE_DIRS}
  ${Tpetra_INCLUDE_DIRS}
  ${Teuchos_INCLUDE_DIRS}
  ${Belos_INCLUDE_DIRS}
  )
target_include_directories(skelly PUBLIC ${SKELLY_INCLUDES})
target_link_libraries(skelly PUBLIC ${STKFMM_LIBRARIES} ${PVFMM_LIB_DIR}/${PVFMM_STATIC_LIB} ${PVFMM_DEP_LIB} OpenMP::OpenMP_CXX)

add_subdirectory(extern/spdlog)
add_subdirectory(extern/trng4)

add_executable(skelly_sim src/skelly_sim.cpp)

target_link_libraries(skelly_sim PUBLIC skelly z MPI::MPI_CXX trng4_static
  ${Kokkos_LIBRARIES} ${Tpetra_LIBRARIES} ${Teuchos_LIBRARIES} ${Belos_LIBRARIES})

include(CTest)
add_subdirectory(tests)
