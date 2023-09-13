#ifndef TESTS_MPI_ENVIRONMENT_HPP_
#define TESTS_MPI_ENVIRONMENT_HPP_

/// \file mpi_environment.hpp
/// \brief Test MPI environment controller

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

/// \class MPIEnvironment
/// \brief MPI environment interface with google test

class MPIEnvironment : public ::testing::Environment {
 public:
  virtual void SetUp() {
    char **argv;
    int argc = 0;
    int mpiError = MPI_Init(&argc, &argv);
    ASSERT_FALSE(mpiError);
  }

  virtual void TearDown() {
    int mpiError = MPI_Finalize();
    ASSERT_FALSE(mpiError);
  }

  virtual ~MPIEnvironment() {
  }
};  // MPIEnvironment

#endif  // TESTS_MPI_ENVIRONMENT_HPP_
