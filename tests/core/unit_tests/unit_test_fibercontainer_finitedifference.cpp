/// \file unit_test_fibercontainer_finitedifference.cpp
/// \brief Unit tests for FiberContainerFinitedifference(single MPI rank)

// C++ includes
#include <iostream>

// skelly includes
#include <fiber_container_finitedifference.hpp>
#include <serialization.hpp>

// test files
#include "./mpi_environment.hpp"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment);

    return RUN_ALL_TESTS();
}

TEST(FiberContainerFiniteDifference, ConstructN1Fibers) {

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

TEST(FiberContainerFiniteDifference, ConstructN10Fibers) {

    // Get a finite difference TOML file
    const std::string toml_file = std::string(TEST_FILES_DIR) + "/fiber_container_fdf_n10.toml";
    auto param_table = toml::parse(toml_file);
    auto params = Params(param_table.at("params"));
    params.print();

    // Construct an abstract fiber container, pointed at a FiberContainerFiniteDifference
    std::unique_ptr<FiberContainerBase> fiber_container;
    fiber_container = std::make_unique<FiberContainerFinitedifference>(param_table.at("fibers").as_array(), params);

    // Test if the number of nodes, etc, in the fiber is what we expect
    EXPECT_EQ(fiber_container->get_local_fiber_number(), 10);
    EXPECT_EQ(fiber_container->get_local_node_count(), 320);
    EXPECT_EQ(fiber_container->get_local_solution_size(), 1280);
    EXPECT_EQ(fiber_container->get_global_fiber_number(), 10);
    EXPECT_EQ(fiber_container->get_global_node_count(), 320);
    EXPECT_EQ(fiber_container->get_global_solution_size(), 1280);
}

TEST(FiberContainerFiniteDifference, SimpleSerializeDeserialize) {

    // Get a finite difference TOML file
    const std::string toml_file = std::string(TEST_FILES_DIR) + "/fiber_container_fdf_n1.toml";
    auto param_table = toml::parse(toml_file);
    auto params = Params(param_table.at("params"));
    params.print();

    // Construct an abstract fiber container, pointed at a FiberContainerFiniteDifference
    std::unique_ptr<FiberContainerBase> fiber_container;
    fiber_container = std::make_unique<FiberContainerFinitedifference>(param_table.at("fibers").as_array(), params);

    // Get a pointer to the derived class
    FiberContainerFinitedifference *fiber_container_fd =
        dynamic_cast<FiberContainerFinitedifference *>(fiber_container.get());

    // Construct the serialization of the derived class
    std::stringstream sbuf;
    msgpack::pack(sbuf, *fiber_container_fd);

    // Now deserialize to get it back out
    std::size_t offset = 0;
    auto const &str = sbuf.str();
    auto oh = msgpack::unpack(str.data(), str.size(), offset);
    auto obj = oh.get();
    // Get a new boject to make sure we aren't accidentally just reading the old one
    FiberContainerFinitedifference fiber_container_deserialized = obj.as<FiberContainerFinitedifference>();

    // Compare the x_ variable inside the fibers for if we read it back in correctly
    for (std::size_t i = 0; i < fiber_container_fd->fibers_.size(); ++i) {
        EXPECT_EQ(fiber_container_fd->fibers_[i].x_, fiber_container_deserialized.fibers_[i].x_);
    }
}

