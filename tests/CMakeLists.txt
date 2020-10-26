set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/tests)
set(CTEST_BINARY_DIRECTORY ${PROJECT_BINARY_DIR}/tests)

file(GLOB files "test_*.cpp")

configure_file("np_fib.npz" "np_fib.npz" COPYONLY)
add_library(cnpy STATIC cnpy.cpp)

foreach(file ${files})
	string(REGEX REPLACE "(^.*/|\\.[^.]*$)" "" file_without_ext ${file})
	add_executable(${file_without_ext} ${file})
	target_link_libraries(${file_without_ext} skelly z OpenMP::OpenMP_CXX MPI::MPI_CXX cnpy
      ${Kokkos_LIBRARIES} ${Tpetra_LIBRARIES} ${Teuchos_LIBRARIES} ${Belos_LIBRARIES})
    target_include_directories(${file_without_ext} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PVFMM_INCLUDE_DIR}/pvfmm ${PVFMM_DEP_INCLUDE_DIR})
	add_test(${file_without_ext} ${file_without_ext})
	set_tests_properties(${file_without_ext}
		PROPERTIES
		PASS_REGULAR_EXPRESSION "Test passed"
        FAIL_REGULAR_EXPRESSION "(Exception|Test failed)"
        TIMEOUT 120
        )

endforeach()
