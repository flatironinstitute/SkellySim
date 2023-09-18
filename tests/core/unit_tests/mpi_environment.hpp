#ifndef TESTS_MPI_ENVIRONMENT_HPP_
#define TESTS_MPI_ENVIRONMENT_HPP_

/// \file mpi_environment.hpp
/// \brief Test MPI environment controller

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

// External includes
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

/// \class MPIEnvironment
/// \brief MPI environment interface with google test

class MPIEnvironment : public ::testing::Environment {
  public:
    virtual void SetUp() {
        char **argv;
        int argc = 0;
        int mpiError = MPI_Init(&argc, &argv);
        ASSERT_FALSE(mpiError);

        // Load the MPI configuration for spdlog for our application
        int mrank;
        int msize;
        MPI_Comm_rank(MPI_COMM_WORLD, &mrank);
        MPI_Comm_size(MPI_COMM_WORLD, &msize);
        spdlog::logger sink =
            mrank == 0 ? spdlog::logger("SkellySim", std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>())
                       : spdlog::logger("SkellySim", std::make_shared<spdlog::sinks::null_sink_st>());
        spdlog::set_default_logger(std::make_shared<spdlog::logger>(sink));
        spdlog::stderr_color_mt("STKFMM");
        spdlog::stderr_color_mt("Belos");
        spdlog::stderr_color_mt("SkellySim global");
        spdlog::cfg::load_env_levels();
    }

    virtual void TearDown() {
        int mpiError = MPI_Finalize();
        ASSERT_FALSE(mpiError);
    }

    virtual ~MPIEnvironment() {}
}; // MPIEnvironment

#endif // TESTS_MPI_ENVIRONMENT_HPP_
