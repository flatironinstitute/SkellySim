/// \file unit_test_fiber_finite_difference.cpp
/// \brief Unit tests for FiberFiniteDifference (single MPI rank)

// C++ includes
#include <iostream>

// skelly includes
#include <body_ellipsoidal.hpp>

// test files
#include "./mpi_environment.hpp"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
    return RUN_ALL_TESTS();
}

TEST(EllipsoidalBody, ConstructorTest) {}
