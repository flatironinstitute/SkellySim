/// \file unit_test_fiber_finite_difference.cpp
/// \brief Unit tests for FiberFiniteDifference (single MPI rank)

// C++ includes
#include <iostream>

// skelly includes
#include <body_container.hpp>
#include <body_ellipsoidal.hpp>

// test files
#include "./mpi_environment.hpp"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
    return RUN_ALL_TESTS();
}

TEST(EllipsoidalBody, ConstructorTest) {

    // Get the ellipsoidal bodies read in
    const std::string toml_file = std::string(TEST_FILES_DIR) + "/ellipsoidal_body_n1.toml";
    auto param_table = toml::parse(toml_file);
    auto params = Params(param_table.at("params"));
    params.print();

    // Construct the body container for this, should handle the abstraction itself
    // FIXME XXX Currently we have issues loading in the quadrature points on the body from a separate directory
    BodyContainer bc = BodyContainer();
}
