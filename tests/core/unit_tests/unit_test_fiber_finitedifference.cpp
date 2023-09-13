/// \file unit_test_configurator.cpp
/// \brief Unit tests for Configurator (single MPI rank)

// C++ includes
#include <iostream>

// skelly includes
#include <fiber_finitedifference.hpp>

// test files
#include "./mpi_environment.hpp"

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}

TEST(FiberFiniteDifference, Compile) {
    //FiberFiniteDifference fiber(8, 1.0, 2.0, 0.5, 0.1);
}
