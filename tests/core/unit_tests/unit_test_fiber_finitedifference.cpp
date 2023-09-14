/// \file unit_test_configurator.cpp
/// \brief Unit tests for Configurator (single MPI rank)

// C++ includes
#include <iostream>

// skelly includes
#include <fiber_finitedifference.hpp>

// test files
#include "./mpi_environment.hpp"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
    return RUN_ALL_TESTS();
}

TEST(FiberFiniteDifference, ConstructorTest) {
    double flength = 1.0;
    double bending_rigidity = 0.0025;
    double radius = 0.0125;
    double eta = 1.0;
    int n_nodes = 32;
    FiberFiniteDifference fiber(n_nodes, radius, flength, bending_rigidity, eta);

    // Do some simple assertions to test if we have the right number of columns, etc
    ASSERT_EQ(3, fiber.x_.rows());
    ASSERT_EQ(n_nodes, fiber.x_.cols());
}
