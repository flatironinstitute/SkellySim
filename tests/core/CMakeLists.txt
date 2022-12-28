
file(GLOB test_src "*.cpp")

foreach(srcfile ${test_src})
  string(REPLACE ".cpp" "" executable ${srcfile})
  get_filename_component(executable ${executable} NAME)
  add_executable(${executable} ${srcfile})
  target_link_libraries(${executable} PUBLIC skelly z MPI::MPI_CXX trng4_static
    ${Kokkos_LIBRARIES} ${Tpetra_LIBRARIES} ${Teuchos_LIBRARIES} ${Belos_LIBRARIES})
endforeach()

foreach(kernel stokeslet;stresslet)
  foreach(driver single;openmp;gpu;fmm)
    add_test(NAME ${kernel}_${driver} COMMAND kernel_test --kernel=${kernel} --driver=${driver})
  endforeach()
endforeach()
