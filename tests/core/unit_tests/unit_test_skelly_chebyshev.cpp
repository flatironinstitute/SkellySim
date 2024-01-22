
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
    EXPECT_EQ(minvratio, 2.0 / 15.0);

    // ChebyshevTPoints
    auto s = ChebyshevTPoints(4);
    EXPECT_THAT(s[0], testing::DoubleEq(-0.9238795325112867));
    EXPECT_THAT(s[1], testing::DoubleEq(-0.3826834323650897));
    EXPECT_THAT(s[2], testing::DoubleEq(0.38268343236508984));
    EXPECT_THAT(s[3], testing::DoubleEq(0.9238795325112867));

    // ChebyshevTPoints in scaled space
    auto s_scaled = ChebyshevTPoints(-10.0, 5.0, 4);
    Eigen::VectorXd s_scaled_julia{{-9.42909649383465, -5.370125742738173, 0.370125742738173, 4.429096493834651}};
    // Use the eigen internal precision functions
    auto is_close = s_scaled.isApprox(s_scaled_julia);
    EXPECT_TRUE(is_close);

    // ChebyshevTPoints in the [0,1] space for comparison with Julia
    auto s5 = ChebyshevTPoints(0.0, 1.0, 5);
    Eigen::VectorXd s5_julia{{0.024471741852423234, 0.2061073738537635, 0.5, 0.7938926261462366, 0.9755282581475768}};
    EXPECT_TRUE(s5.isApprox(s5_julia));
}

// Test Vandermonde matrix construction (including from Julia code)
TEST(SkellyChebyshev, vandermonde) {
    // Get a vandermonde matrix for chebyshev coeffs
    Eigen::VectorXd s = ChebyshevTPoints(4);
    Eigen::MatrixXd A = vander_julia_chebyshev(s, 4 - 1);
    // Check against Julia
    Eigen::VectorXd j0{{1.0, 1.0, 1.0, 1.0}};
    Eigen::VectorXd j1{{-0.9238795325112867, -0.3826834323650897, 0.38268343236508984, 0.9238795325112867}};
    Eigen::VectorXd j2{{0.7071067811865475, -0.7071067811865476, -0.7071067811865475, 0.7071067811865475}};
    Eigen::VectorXd j3{{-0.3826834323650896, 0.9238795325112867, -0.9238795325112868, 0.3826834323650896}};
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
    Eigen::VectorXd i0{
        {0.2000000000000002, -0.3804226065180616, 0.3236067977499791, -0.2351141009169892, 0.12360679774997901}};
    EXPECT_TRUE(IVM.col(0).isApprox(i0));
}

// Test derivative functions
TEST(SkellyChebyshev, derivatives) {
    // Test the julia derivative function
    Eigen::VectorXd p{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    Eigen::VectorXd dp = derivative_julia_chebyshev(p);
    Eigen::VectorXd dp_true{{7.0, 0.0, 14.0, 0.0, 14.0, 0.0, 14.0, 0.0}};
    EXPECT_TRUE(dp.isApprox(dp_true));

    // Try multiple derivatives for 5 points
    Eigen::VectorXd d1_5 = NthDerivativeOfChebyshevTn(5, 1);
    Eigen::VectorXd d2_5 = NthDerivativeOfChebyshevTn(5, 2);
    Eigen::VectorXd d1_5_true{{5.0, 0.0, 10.0, 0.0, 10.0}};
    Eigen::VectorXd d2_5_true{{0.0, 120.0, 0.0, 80.0}};
    EXPECT_TRUE(d1_5.isApprox(d1_5_true));
    EXPECT_TRUE(d2_5.isApprox(d2_5_true));

    // Try the derivative matrix
    Eigen::MatrixXd D1 = DerivativeMatrix(8, 1);
    Eigen::MatrixXd D2 = DerivativeMatrix(8, 2);
    Eigen::MatrixXd D3 = DerivativeMatrix(9, 3);
    Eigen::MatrixXd D1_true{{0.0, 1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0},   {0.0, 0.0, 4.0, 0.0, 8.0, 0.0, 12.0, 0.0},
                            {0.0, 0.0, 0.0, 6.0, 0.0, 10.0, 0.0, 14.0}, {0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 12.0, 0.0},
                            {0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 14.0}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0},
                            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0}};
    Eigen::MatrixXd D2_true{{0.0, 0.0, 4.0, 0.0, 32.0, 0.0, 108.0, 0.0}, {0.0, 0.0, 0.0, 24.0, 0.0, 120.0, 0.0, 336.0},
                            {0.0, 0.0, 0.0, 0.0, 48.0, 0.0, 192.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0, 80.0, 0.0, 280.0},
                            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 120.0, 0.0},  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 168.0}};
    Eigen::MatrixXd D3_true{
        {0.0, 0.0, 0.0, 24.0, 0.0, 360.0, 0.0, 2016.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 192.0, 0.0, 1728.0, 0.0, 7680.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 480.0, 0.0, 3360.0, 0.0},  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 960.0, 0.0, 5760.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1680.0, 0.0},    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2688.0}};
    EXPECT_TRUE(D1.isApprox(D1_true));
    EXPECT_TRUE(D2.isApprox(D2_true));
    EXPECT_TRUE(D3.isApprox(D3_true));
}

// Test integral matrix
TEST(SkellyChebyshev, integrals) {
    // Try a single integral matrix
    Eigen::MatrixXd i8 = IntegrationMatrix(8);
    Eigen::MatrixXd i8_true{{1.0000000000000000, -0.2500000000000000, -0.3333333333333333, 0.1250000000000000,
                             -0.0666666666666667, 0.0416666666666667, -0.0285714285714286, 1.0000000000000000},
                            {1.0000000000000000, 0.0000000000000000, -0.5000000000000000, 0.0000000000000000,
                             0.0000000000000000, 0.0000000000000000, -0.0000000000000000, 0.0000000000000000},
                            {0.0000000000000000, 0.2500000000000000, 0.0000000000000000, -0.2500000000000000,
                             0.0000000000000000, 0.0000000000000000, -0.0000000000000000, 0.0000000000000000},
                            {0.0000000000000000, 0.0000000000000000, 0.1666666666666667, 0.0000000000000000,
                             -0.1666666666666667, -0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
                            {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.1250000000000000,
                             0.0000000000000000, -0.1250000000000000, -0.0000000000000000, 0.0000000000000000},
                            {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                             0.1000000000000000, 0.0000000000000000, -0.1000000000000000, 0.0000000000000000},
                            {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                             0.0000000000000000, 0.0833333333333333, -0.0000000000000000, 0.0000000000000000},
                            {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                             0.0000000000000000, 0.0000000000000000, 0.0714285714285714, 0.0000000000000000}};
    EXPECT_TRUE(i8.isApprox(i8_true));
}
