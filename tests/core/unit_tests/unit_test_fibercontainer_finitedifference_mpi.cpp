/// \file unit_test_fibercontainer_finitedifference_mpi.cpp
/// \brief Unit tests for FiberContainerFinitedifference(2 MPI ranks)

// C++ includes
#include <iostream>

// skelly includes
#include <fiber_container_finitedifference.hpp>

// test files
#include "./mpi_environment.hpp"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment);

    return RUN_ALL_TESTS();
}

TEST(FiberContainerFiniteDifference, MPIConstructN10Fibers) {

    // Get a finite difference TOML file
    const std::string toml_file = std::string(TEST_FILES_DIR) + "/fiber_container_fdf_n10.toml";
    auto param_table = toml::parse(toml_file);
    auto params = Params(param_table.at("params"));
    params.print();

    // Construct an abstract fiber container, pointed at a FiberContainerFiniteDifference
    std::unique_ptr<FiberContainerBase> fiber_container;
    fiber_container = std::make_unique<FiberContainerFinitedifference>(param_table.at("fibers").as_array(), params);

    // Test if the number of nodes, etc, in the fiber is what we expect
    EXPECT_EQ(fiber_container->get_local_fiber_number(), 5);
    EXPECT_EQ(fiber_container->get_local_node_count(), 160);
    EXPECT_EQ(fiber_container->get_local_solution_size(), 640);
    EXPECT_EQ(fiber_container->get_global_fiber_number(), 10);
    EXPECT_EQ(fiber_container->get_global_node_count(), 320);
    EXPECT_EQ(fiber_container->get_global_solution_size(), 1280);
}

