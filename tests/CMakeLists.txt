set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/tests)
set(CTEST_BINARY_DIRECTORY ${PROJECT_BINARY_DIR}/tests)

file(GLOB files "test_*.cpp")

configure_file("np_fib.npz" "np_fib.npz" COPYONLY)

foreach(file ${files})
	string(REGEX REPLACE "(^.*/|\\.[^.]*$)" "" file_without_ext ${file})
	add_executable(${file_without_ext} ${file} cnpy.cpp)
	target_link_libraries(${file_without_ext} skelly z)
    target_include_directories(${file_without_ext} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PVFMM_INCLUDE_DIR}/pvfmm ${PVFMM_DEP_INCLUDE_DIR})
	add_test(${file_without_ext} ${file_without_ext})
	set_tests_properties(${file_without_ext}
		PROPERTIES
		PASS_REGULAR_EXPRESSION "Test passed")
	set_tests_properties(${file_without_ext}
		PROPERTIES
		FAIL_REGULAR_EXPRESSION "(Exception|Test failed)")
	set_tests_properties(${file_without_ext}
		PROPERTIES
		TIMEOUT 120)
endforeach()
