
set(TEST_LIST
  unit_test_fiber_finite_difference
  unit_test_fibercontainer_finite_difference
  unit_test_serialization
  unit_test_body_ellipsoidal
  unit_test_skelly_chebyshev
)

# Setup the MPI testing environment for the MPI tests
# Taken from hoomd-blue v4.0.1
MACRO(ADD_TO_MPI_TESTS _KEY _VALUE)
SET("NProc_${_KEY}" "${_VALUE}")
SET(MPI_TEST_LIST ${MPI_TEST_LIST} ${_KEY})
ENDMACRO(ADD_TO_MPI_TESTS)

ADD_TO_MPI_TESTS(unit_test_fibercontainer_finite_difference_mpi 2)
ADD_TO_MPI_TESTS(unit_test_serialization_mpi 2)

# Set up the test executables
foreach (CUR_TEST ${TEST_LIST} ${MPI_TEST_LIST})
  # Add the executable
  add_executable(${CUR_TEST} EXCLUDE_FROM_ALL ${CUR_TEST}.cpp)

  target_compile_definitions(${CUR_TEST} PRIVATE TEST_FILES_DIR="${TEST_FILES_DIR}")

  add_dependencies(test_all ${CUR_TEST})

  target_link_libraries(${CUR_TEST} PRIVATE
    skelly
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
    GTest::gmock_main)
  target_include_directories(${CUR_TEST} PRIVATE
    "${googletest_SOURCE_DIR}/googlemock/include"
    "${googletest_SOURCE_DIR}/googletest/include")
endforeach (CUR_TEST)

# Add the non-mpi tests first (1 processor)
foreach (CUR_TEST ${TEST_LIST})
  add_test(NAME ${CUR_TEST} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_POSTFLAGS} $<TARGET_FILE:${CUR_TEST}>)
endforeach (CUR_TEST)

# Add the mpi tests (variable processors)
foreach (CUR_TEST ${MPI_TEST_LIST})
  set(MPI_TEST_NAME mpi-${CUR_TEST})

  add_test(NAME ${MPI_TEST_NAME} COMMAND
    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}
    ${NProc_${CUR_TEST}} ${MPIEXEC_POSTFLAGS}
    $<TARGET_FILE:${CUR_TEST}>)
endforeach (CUR_TEST)
