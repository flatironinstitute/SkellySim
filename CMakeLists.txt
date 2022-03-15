cmake_minimum_required(VERSION 3.10)

# https://stackoverflow.com/questions/61499646/cmake-set-variable-readonly-protect-from-override
# need this for pvfmm, since pvfmmConfig.cmake hard codes install paths, which STKFMM would then use
macro(set_readonly VAR)
  # Set the variable itself
  set("${VAR}" "${ARGN}")
  # Store the variable's value for restore it upon modifications.
  set("_${VAR}_readonly_val" "${ARGN}")
  # Register a watcher for a variable
  variable_watch("${VAR}" readonly_guard)
endmacro()

# Watcher for a variable which emulates readonly property.
macro(readonly_guard VAR access value current_list_file stack)
  if ("${access}" STREQUAL "MODIFIED_ACCESS")
    # Restore a value of the variable to the initial one.
    set(${VAR} "${_${VAR}_readonly_val}")
  endif()
endmacro()


# Get the current git tag
execute_process(
  COMMAND git describe --tags --abbrev=0
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_TAG
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "v" "" GIT_TAG ${GIT_TAG})

# Get the current git commit
execute_process(
  COMMAND git describe --abbrev=8 --dirty --always
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message("SkellySim version: v${GIT_TAG} (${GIT_COMMIT})")

project(
  skelly_sim
  VERSION ${GIT_TAG}
  LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
option(EIGEN_USE_BLAS "Use BLAS as the backend for various Eigen calls" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
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

add_definitions("-DGIT_TAG=\"${GIT_TAG}\"")
add_definitions("-DGIT_COMMIT=\"${GIT_COMMIT}\"")

# required compiler features
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Kokkos REQUIRED HINTS $ENV{TRILINOS_BASE}/lib/cmake)
find_package(Tpetra REQUIRED HINTS $ENV{TRILINOS_BASE}/lib/cmake)
find_package(Teuchos REQUIRED HINTS $ENV{TRILINOS_BASE}/lib/cmake)
find_package(Belos REQUIRED HINTS $ENV{TRILINOS_BASE}/lib/cmake)

add_library(skelly OBJECT src/fiber.cpp src/kernels.cpp src/utils.cpp src/periphery.cpp src/cnpy.cpp src/params.cpp
  src/system.cpp src/body.cpp src/solver_hydro.cpp src/rng.cpp src/spherical_body.cpp src/deformable_body.cpp src/point_source.cpp)

if(SKELLY_BUILD_FMM)
  set(pvfmm_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern/pvfmm)
  add_subdirectory(extern/pvfmm EXCLUDE_FROM_ALL)

  set_readonly(PVFMM_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/extern/pvfmm/lib64/")
  set_readonly(PVFMM_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/pvfmm-include/")
  find_package(pvfmm)
  add_subdirectory(extern/stkfmm EXCLUDE_FROM_ALL)

  file(GLOB pvfmm_includes "${PROJECT_SOURCE_DIR}/extern/pvfmm/include/*")

  add_custom_target(pvfmm_include)
  add_custom_command(TARGET pvfmm_include
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/extern/pvfmm/pvfmm_config.h pvfmm-include/pvfmm/pvfmm_config.h
    )

  foreach(fullsource ${pvfmm_includes})
    get_filename_component(source ${fullsource} NAME)
    add_custom_command(TARGET pvfmm_include
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${fullsource} pvfmm-include/pvfmm/${source}
      )
  endforeach()

  file(GLOB sctl_includes "${PROJECT_SOURCE_DIR}/extern/pvfmm/SCTL/include/sctl/*")
  add_custom_command(TARGET pvfmm_include
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${PROJECT_SOURCE_DIR}/extern/pvfmm/SCTL/include/sctl.hpp pvfmm-include/pvfmm/sctl.hpp
    )
  foreach(fullsource ${sctl_includes})
    get_filename_component(source ${fullsource} NAME)
    add_custom_command(TARGET pvfmm_include
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${fullsource} pvfmm-include/pvfmm/sctl/${source}
      )
  endforeach()

  add_dependencies(STKFMM_STATIC pvfmm_include pvfmmStatic)
  add_dependencies(skelly STKFMM_STATIC)
  # FIXME: Lazy hack to load mkl fftw if loaded
  set(SKELLY_INCLUDES_FMM ${PVFMM_INCLUDE_DIR}/pvfmm
    ${PROJECT_SOURCE_DIR}/extern/stkfmm/Lib/include)
  if (MKL_FFTW_INCLUDE_DIR)
    set(SKELLY_INCLUDES_FMM ${SKELLY_INCLUDES_FMM} ${MKL_FFTW_INCLUDE_DIR})
  endif()
  target_link_libraries(skelly PUBLIC STKFMM_STATIC.a pvfmm.a ${PVFMM_DEP_LIB} OpenMP::OpenMP_CXX)
  target_link_directories(skelly PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/extern/stkfmm/Lib ${CMAKE_CURRENT_BINARY_DIR}/extern/pvfmm)
else()
  find_package(pvfmm REQUIRED)
  find_package(STKFMM REQUIRED)
  set(SKELLY_INCLUDES_FMM ${PVFMM_INCLUDE_DIR}/pvfmm
    ${PVFMM_DEP_INCLUDE_DIR}
    ${STKFMM_INCLUDE_DIRS}
  ) 
  target_link_libraries(skelly PUBLIC ${STKFMM_LIBRARIES} ${PVFMM_LIB_DIR}/${PVFMM_STATIC_LIB} ${PVFMM_DEP_LIB} OpenMP::OpenMP_CXX)
endif()


set(SKELLY_INCLUDES ${PROJECT_SOURCE_DIR}/include)
set(SKELLY_INCLUDES_SYSTEM
  ${PROJECT_SOURCE_DIR}/extern/spdlog/include
  ${PROJECT_SOURCE_DIR}/extern/trng4
  ${PROJECT_SOURCE_DIR}/extern/toml11
  ${PROJECT_SOURCE_DIR}/extern/msgpack-c/include
  ${MPI_CXX_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${SKELLY_INCLUDES_FMM}
  ${Kokkos_INCLUDE_DIRS}
  ${Tpetra_INCLUDE_DIRS}
  ${Teuchos_INCLUDE_DIRS}
  ${Belos_INCLUDE_DIRS}
  )
target_include_directories(skelly PUBLIC ${SKELLY_INCLUDES} SYSTEM ${SKELLY_INCLUDES_SYSTEM})

add_subdirectory(extern/spdlog EXCLUDE_FROM_ALL)
add_subdirectory(extern/trng4 EXCLUDE_FROM_ALL)

add_executable(skelly_sim src/skelly_sim.cpp)

target_link_libraries(skelly_sim PUBLIC skelly z MPI::MPI_CXX trng4_static
  ${Kokkos_LIBRARIES} ${Tpetra_LIBRARIES} ${Teuchos_LIBRARIES} ${Belos_LIBRARIES})


add_executable(skelly_debug src/skelly_debug.cpp)
target_link_libraries(skelly_debug PUBLIC skelly z MPI::MPI_CXX trng4_static
  ${Kokkos_LIBRARIES} ${Tpetra_LIBRARIES} ${Teuchos_LIBRARIES} ${Belos_LIBRARIES})


include(CTest)
add_subdirectory(tests)
configure_file(module_template.in module)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/paraview
  DESTINATION share/skelly_sim
  PATTERN "__pycache__" EXCLUDE)

install(TARGETS skelly_sim
  CONFIGURATIONS Release
  RUNTIME DESTINATION bin)
