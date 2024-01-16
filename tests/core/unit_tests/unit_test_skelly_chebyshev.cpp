
/// \file unit_test_serialization.cpp
/// \brief Unit tests for serialization via msgpack (single MPI rank)

// C++ includes
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// skelly includes
#include <msgpack.hpp>
#include <skelly_chebyshev.hpp>

// test files
#include "./mpi_environment.hpp"

// Using command for getting into the correct namespace for everything
using namespace skelly_chebyshev;

// Test if the Chebyshev convenience functions work correctly
TEST(SkellyChebyshev, chevyshev_convenience) {
    // Ratio
    double mratio = chebyshev_ratio(-10.0, 10.0);
    EXPECT_EQ(mratio, 10.0);

    // Inverse ratio
    double minvratio = inverse_chebyshev_ratio(-10.0, 5.0);
    EXPECT_EQ(minvratio, 2.0/15.0);

    // ChebyshevTPoints
    auto s = ChebyshevTPoints(4);
    EXPECT_THAT(s[0], testing::DoubleEq(-0.9238795325112867));
    EXPECT_THAT(s[1], testing::DoubleEq(-0.3826834323650897));
    EXPECT_THAT(s[2], testing::DoubleEq(0.38268343236508984));
    EXPECT_THAT(s[3], testing::DoubleEq(0.9238795325112867));

    // ChebyshevTPoints in scaled space
    auto s_scaled = ChebyshevTPoints(-10.0, 5.0, 4);
    Eigen::VectorXd s_scaled_julia {{-9.42909649383465, -5.370125742738173, 0.370125742738173, 4.429096493834651}};
    // Use the eigen internal precision functions
    auto is_close = s_scaled.isApprox(s_scaled_julia);
    EXPECT_TRUE(is_close);

    // ChebyshevTPoints in the [0,1] space for comparison with Julia
    auto s5 = ChebyshevTPoints(0.0, 1.0, 5);
    Eigen::VectorXd s5_julia {{0.024471741852423234, 0.2061073738537635, 0.5, 0.7938926261462366, 0.9755282581475768}};
    EXPECT_TRUE(s5.isApprox(s5_julia));
}

// Test Vandermonde matrix construction (including from Julia code)
TEST(SkellyChebyshev, vandermonde) {
    // Get a vandermonde matrix for chebyshev coeffs
    Eigen::VectorXd s = ChebyshevTPoints(4);
    Eigen::MatrixXd A = vander_julia_chebyshev(s, 4-1);
    // Check against Julia
    Eigen::VectorXd j0 {{1.0, 1.0, 1.0, 1.0}};
    Eigen::VectorXd j1 {{-0.9238795325112867, -0.3826834323650897, 0.38268343236508984, 0.9238795325112867}};
    Eigen::VectorXd j2 {{0.7071067811865475, -0.7071067811865476, -0.7071067811865475, 0.7071067811865475}};
    Eigen::VectorXd j3 {{-0.3826834323650896, 0.9238795325112867, -0.9238795325112868, 0.3826834323650896}};
    auto A0 = A.col(0);
    auto A1 = A.col(1);
    auto A2 = A.col(2);
    auto A3 = A.col(3);
    EXPECT_TRUE(A0.isApprox(j0));
    EXPECT_TRUE(A1.isApprox(j1));
    EXPECT_TRUE(A2.isApprox(j2));
    EXPECT_TRUE(A3.isApprox(j3));

    // Try the order-ed vandermonde matrix
    Eigen::MatrixXd VM = VandermondeMatrix(4);
    EXPECT_TRUE(VM.col(0).isApprox(j0));
    EXPECT_TRUE(VM.col(1).isApprox(j1));
    EXPECT_TRUE(VM.col(2).isApprox(j2));
    EXPECT_TRUE(VM.col(3).isApprox(j3));

    // Try the inverse vandermonde matrix
    Eigen::MatrixXd IVM = InverseVandermondeMatrix(5);
    Eigen::VectorXd i0 {{0.2000000000000002, -0.3804226065180616, 0.3236067977499791, -0.2351141009169892, 0.12360679774997901}};
    EXPECT_TRUE(IVM.col(0).isApprox(i0));

    // Try to toggle the representation
    // XXX CJE come back to here once I have all of the vandermonde matrices
}
