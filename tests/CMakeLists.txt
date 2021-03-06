set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/tests)
set(CTEST_BINARY_DIRECTORY ${PROJECT_BINARY_DIR}/tests)

file(GLOB files "test_*.cpp")

configure_file("np_fib.npz" "np_fib.npz" COPYONLY)
configure_file("2K_MTs_onCortex_R5_L1.toml" "2K_MTs_onCortex_R5_L1.toml" COPYONLY)
configure_file("test_body.toml" "test_body.toml" COPYONLY)
configure_file("test_gmres.toml" "test_gmres.toml" COPYONLY)

add_test(NAME "make_precompute_data_periphery" COMMAND "python3" "${CMAKE_SOURCE_DIR}/utils/make_precompute_data.py" "test_periphery.toml")
add_test(NAME "make_precompute_data_body" COMMAND "python3" "${CMAKE_SOURCE_DIR}/utils/make_precompute_data.py" "test_body.toml")
add_test(NAME "make_precompute_data_gmres" COMMAND "python3" "${CMAKE_SOURCE_DIR}/utils/make_precompute_data.py" "test_gmres.toml")

foreach(file ${files})
  string(REGEX REPLACE "(^.*/|\\.[^.]*$)" "" file_without_ext ${file})
  add_executable(${file_without_ext} ${file})

  target_link_libraries(${file_without_ext} skelly z OpenMP::OpenMP_CXX MPI::MPI_CXX trng4_static
    ${Kokkos_LIBRARIES} ${Tpetra_LIBRARIES} ${Teuchos_LIBRARIES} ${Belos_LIBRARIES})
  
  target_include_directories(${file_without_ext} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/extern/spdlog/include
    ${PROJECT_SOURCE_DIR}/extern/toml11
    ${PROJECT_SOURCE_DIR}/extern/msgpack-c/include
    ${PVFMM_INCLUDE_DIR}/pvfmm
    ${PVFMM_DEP_INCLUDE_DIR}
    )

  add_test(${file_without_ext} ${file_without_ext})

  set_tests_properties(${file_without_ext}
    PROPERTIES
    PASS_REGULAR_EXPRESSION "Test passed"
    FAIL_REGULAR_EXPRESSION "(Exception|Test failed)"
    TIMEOUT 120
    )

endforeach()
