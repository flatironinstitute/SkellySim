/// \file unit_test_fibercontainer_finitedifference.cpp
/// \brief Unit tests for FiberContainerFinitedifference(single MPI rank)

// C++ includes
#include <iostream>

// skelly includes
#include <fiber_container_finitedifference.hpp>

// External includes
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

// test files
#include "./mpi_environment.hpp"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
    return RUN_ALL_TESTS();
}

TEST(FiberContainerFiniteDifference, ConstructorTest) {
    // Load the MPI configuration for ourselves
    int mrank;
    int msize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mrank);
    MPI_Comm_size(MPI_COMM_WORLD, &msize);
    spdlog::logger sink =
        mrank == 0 ? spdlog::logger("SkellySimTest", std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>())
                   : spdlog::logger("SkellySimTest", std::make_shared<spdlog::sinks::null_sink_st>());
    spdlog::set_default_logger(std::make_shared<spdlog::logger>(sink));
    spdlog::stderr_color_mt("STKFMM");
    spdlog::stderr_color_mt("Belos");
    spdlog::stderr_color_mt("SkellySimTest global");
    spdlog::cfg::load_env_levels();

    // Get a finite difference TOML file
    const std::string toml_file = std::string(TEST_FILES_DIR) + "/fiber_container_fdf_n1.toml";
    auto param_table = toml::parse(toml_file);
    auto params = Params(param_table.at("params"));
    params.print();

    // Construct an abstract fiber container, pointed at a FiberContainerFiniteDifference
    std::unique_ptr<FiberContainerBase> fiber_container;
    fiber_container = std::make_unique<FiberContainerFinitedifference>(param_table.at("fibers").as_array(), params);

    // Test if the number of nodes, etc, in the fiber is what we expect
    EXPECT_EQ(fiber_container->get_local_fiber_number(), 1);
    EXPECT_EQ(fiber_container->get_local_node_count(), 32);
    EXPECT_EQ(fiber_container->get_local_solution_size(), 128);
    EXPECT_EQ(fiber_container->get_global_fiber_number(), 1);
    EXPECT_EQ(fiber_container->get_global_node_count(), 32);
    EXPECT_EQ(fiber_container->get_global_solution_size(), 128);
}
