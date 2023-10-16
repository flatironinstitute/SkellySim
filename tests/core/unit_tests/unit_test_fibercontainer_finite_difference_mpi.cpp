/// \file unit_test_fibercontainer_finitedifference_mpi.cpp
/// \brief Unit tests for FiberContainerFiniteDifference(2 MPI ranks)

// C++ includes
#include <iostream>

// skelly includes
#include <fiber_container_finite_difference.hpp>
#include <serialization.hpp>

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
    fiber_container = std::make_unique<FiberContainerFiniteDifference>(param_table.at("fibers").as_array(), params);

    // Test if the number of nodes, etc, in the fiber is what we expect
    EXPECT_EQ(fiber_container->get_local_fiber_count(), 5);
    EXPECT_EQ(fiber_container->get_local_node_count(), 160);
    EXPECT_EQ(fiber_container->get_local_solution_size(), 640);
    EXPECT_EQ(fiber_container->get_global_fiber_count(), 10);
    EXPECT_EQ(fiber_container->get_global_node_count(), 320);
    EXPECT_EQ(fiber_container->get_global_solution_size(), 1280);
}

// Create a test input and output map to work with serialization concepts
typedef struct test_input_map_t {
    double dt;
    std::unique_ptr<FiberContainerBase> fibers;
    MSGPACK_DEFINE_MAP(dt, fibers);
} test_input_map_t;

typedef struct test_output_map_t {
    double &dt;
    std::unique_ptr<FiberContainerBase> &fibers;
    MSGPACK_DEFINE_MAP(dt, fibers);
} test_output_map_t;

// This test should go through a full setup of the FiberContainerFiniteDifference, and then also go through how to
// serialize and deserialize it like the write function in system.cpp does.
TEST(FiberContainerFiniteDifference, FullSerializeDeserialize) {

    // Get the MPI environment
    int mrank = 0;
    int msize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mrank);
    MPI_Comm_size(MPI_COMM_WORLD, &msize);

    // Get a finite difference TOML file
    const std::string toml_file = std::string(TEST_FILES_DIR) + "/fiber_container_fdf_n10.toml";
    auto param_table = toml::parse(toml_file);
    auto params = Params(param_table.at("params"));
    params.print();

    // Create the fiber container
    std::unique_ptr<FiberContainerBase> fiber_container;
    fiber_container = std::make_unique<FiberContainerFiniteDifference>(param_table.at("fibers").as_array(), params);

    // Do this exactly as the write function
    // We have to create a global version of the fibers that is the same type as the fibers we already have
    std::unique_ptr<FiberContainerBase> fc_global;
    // Create an empty global fiber container of the correct type
    if (fiber_container->fiber_type_ == FiberContainerBase::FIBERTYPE::FiniteDifference) {
        fc_global = std::make_unique<FiberContainerFiniteDifference>();
    } else {
        throw std::runtime_error("Somehow got an incorrect fiber container base type in test");
    }

    double dt = 0.005;
    const test_output_map_t to_merge{dt, fiber_container};

    std::stringstream mergebuf;
    msgpack::pack(mergebuf, to_merge);

    std::string msg_local = mergebuf.str();
    int msgsize_local = msg_local.size();
    std::vector<int> msgsize(msize);
    MPI_Gather(&msgsize_local, 1, MPI_INT, msgsize.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    int msgsize_global = 0;
    std::vector<int> displs(msize + 1);
    for (int i = 0; i < msize; ++i) {
        msgsize_global += msgsize[i];
        displs[i + 1] = displs[i] + msgsize[i];
    }

    std::vector<uint8_t> msg = (mrank == 0) ? std::vector<uint8_t>(msgsize_global) : std::vector<uint8_t>();
    MPI_Gatherv(msg_local.data(), msgsize_local, MPI_CHAR, msg.data(), msgsize.data(), displs.data(), MPI_CHAR, 0,
                MPI_COMM_WORLD);

    // Try the deserialize
    if (mrank == 0) {
        msgpack::object_handle oh;
        std::size_t offset = 0;

        // Get our test_output_map
        test_output_map_t to_write{dt, fc_global};

        for (int i = 0; i < msize; ++i) {
            msgpack::unpack(oh, (char *)msg.data(), msg.size(), offset);
            msgpack::object obj = oh.get();
            test_input_map_t const &min_state = obj.as<test_input_map_t>();

            // Do a check here for what fiber type, for completeness
            if (min_state.fibers->fiber_type_ == FiberContainerBase::FIBERTYPE::FiniteDifference) {
                // Cast to correct type and do the fibers
                const FiberContainerFiniteDifference *fibers_fd =
                    static_cast<const FiberContainerFiniteDifference *>(min_state.fibers.get());
                // Also need to cast the global fibers to the correct type to use them
                FiberContainerFiniteDifference *fc_fd_global =
                    static_cast<FiberContainerFiniteDifference *>(fc_global.get());
                for (const auto &min_fib : fibers_fd->fibers_) {
                    fc_fd_global->fibers_.emplace_back(FiberFiniteDifference(min_fib, params.eta));
                }
            }
        }

        // Check the contents of the global fiber against what we have
        if (fc_global->fiber_type_ == FiberContainerBase::FIBERTYPE::FiniteDifference) {
            // Cast the global fiber type correctly
            auto fc_fd_global = static_cast<FiberContainerFiniteDifference *>(fc_global.get());
            EXPECT_EQ(fc_fd_global->fibers_.size(), 10);
        }

        std::stringstream output_sbuf;
        msgpack::pack(output_sbuf, to_write);

        // Immediately deserialize the buffer to see what we have inside and check the contents against what we have
        // originally. Ignore the multiple fiber types at the moment, since we know what we are going to be serializing
        // and deserializing.
        std::size_t my_offset = 0;
        auto const &test_string = output_sbuf.str();
        auto test_oh = msgpack::unpack(test_string.data(), test_string.size(), my_offset);
        auto test_obj = test_oh.get();
        // Do full deserialization, unfortunatley
        test_input_map_t const &deserialized_state = test_obj.as<test_input_map_t>();
        const FiberContainerFiniteDifference *fibers_deserialized =
            static_cast<const FiberContainerFiniteDifference *>(deserialized_state.fibers.get());
        const FiberContainerFiniteDifference *fibers_original =
            static_cast<const FiberContainerFiniteDifference *>(fiber_container.get());
        std::cout << "Full fiber deserialization\n";
        for (int i = 0; i < fibers_deserialized->fibers_.size(); ++i) {
            std::cout << fibers_deserialized->fibers_[i].x_ << std::endl;
        }
        std::cout << "rank=0 fiber originals\n";
        for (int i = 0; i < fibers_original->fibers_.size(); ++i) {
            std::cout << fibers_original->fibers_[i].x_ << std::endl;
        }
        for (int i = 0; i < fibers_original->fibers_.size(); ++i) {
            EXPECT_EQ(fibers_original->fibers_[i].x_, fibers_deserialized->fibers_[i].x_);
        }
    }
}
