/// \file unit_test_skelly_chebyshev.cpp
/// \brief Unit tests for Chebyshev polynomial helper functions

// C++ includes
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// skelly includes
#include <msgpack.hpp>
#include <skelly_chebyshev.hpp>

// External includes
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

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

// Test vandermonde matrix cache
TEST(SkellyChebyshev, vandermonde_cache) {
    auto vm4 = getVM(4);
    auto vm8 = getVM(8);
    auto VM4 = VandermondeMatrix(4);
    auto VM8 = VandermondeMatrix(8);

    EXPECT_TRUE(VM4.isApprox(vm4));
    EXPECT_TRUE(VM8.isApprox(vm8));

    auto ivm4 = getIVM(4);
    auto ivm8 = getIVM(8);
    auto IVM4 = InverseVandermondeMatrix(4);
    auto IVM8 = InverseVandermondeMatrix(8);

    EXPECT_TRUE(IVM4.isApprox(ivm4));
    EXPECT_TRUE(IVM8.isApprox(ivm8));
}

// Test conversion of vectors to and from coefficient and node space
TEST(SkellyChebyshev, c2f_f2c) {
    // Try with auto
    auto s4 = ChebyshevTPoints(4);
    auto vm4 = VandermondeMatrix(4);
    auto c2f4_true = vm4 * s4;
    auto c2f4 = C2F(s4, vm4);
    EXPECT_TRUE(c2f4_true.isApprox(c2f4));

    // Try the one that just automatically gets the cached version
    autodiff::VectorXreal s8 = ChebyshevTPoints(8);
    auto vm8 = VandermondeMatrix(8);
    autodiff::VectorXreal c2f8_true = vm8 * s8;
    autodiff::VectorXreal c2f8 = C2F(s8);
    EXPECT_TRUE(c2f8_true.isApprox(c2f8));

    // Try with the inverse operation with a larger array
    autodiff::VectorXdual s40 = ChebyshevTPoints(40);
    auto ivm40 = InverseVandermondeMatrix(40);
    autodiff::VectorXdual f2c40_true = ivm40 * s40;
    autodiff::VectorXdual f2c40 = F2C(s40);
    EXPECT_TRUE(f2c40_true.isApprox(f2c40));
}

// Test toggle representation on a vector
TEST(SkellyChebyshev, toggle_representation_vector) {
    autodiff::VectorXreal s = Eigen::VectorXd::Ones(20);

    // Try toggling to itself
    auto s_nochange = ToggleRepresentation(s, REPR::c, REPR::c);
    // Toggling from coefficient to node
    auto s_c2f = ToggleRepresentation(s, REPR::c, REPR::n);
    // Toggling from node to coefficient
    auto s_f2c = ToggleRepresentation(s, REPR::n, REPR::c);

    autodiff::VectorXreal c2f_real{{0.4803549464961659,  0.5591788998203820, 0.4005438163101717,  0.6410145841510756,
                                    0.3155402614450867,  0.7305031572136578, 0.2199865457629623,  0.8340893189596487,
                                    0.1058317827074546,  0.9621952458291025, -0.0408969526537215, 1.1342469763726617,
                                    -0.2483028813327439, 1.3928142423769696, -0.5845838426771245, 1.8553093046674527,
                                    -1.2728662712798653, 3.0136697460629245, -3.7244786699107979, 13.2258497896785379}};
    autodiff::VectorXreal f2c_real{{1.0000000000000000,  0.0000000000000002,  0.0000000000000000,  -0.0000000000000001,
                                    0.0000000000000000,  0.0000000000000000,  -0.0000000000000000, -0.0000000000000000,
                                    -0.0000000000000000, 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
                                    0.0000000000000000,  0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
                                    -0.0000000000000000, -0.0000000000000000, -0.0000000000000000, 0.0000000000000001}};

    EXPECT_TRUE(s.isApprox(s_nochange));
    EXPECT_TRUE(c2f_real.isApprox(s_c2f));
    EXPECT_TRUE(f2c_real.isApprox(s_f2c));
}

// Test resize of vectors
TEST(SkellyChebyshev, resize) {
    // Create a vector that we can inspect pretty easily
    autodiff::VectorXreal X{{1, 2, 3, 4}};
    autodiff::VectorXreal XC3 = Resize(X, 3, REPR::c, REPR::c);
    autodiff::VectorXreal XC4 = Resize(X, 4, REPR::c, REPR::c);
    autodiff::VectorXreal XC8 = Resize(X, 8, REPR::c, REPR::c);
    autodiff::VectorXreal XC3true{{1, 2, 3}};
    autodiff::VectorXreal XC8true{{1, 2, 3, 4, 0, 0, 0, 0}};

    EXPECT_TRUE(XC3.isApprox(XC3true));
    EXPECT_TRUE(XC4.isApprox(X));
    EXPECT_TRUE(XC8.isApprox(XC8true));

    autodiff::VectorXreal XF4 = Resize(X, 4, REPR::n, REPR::n);
    autodiff::VectorXreal XF8 = Resize(X, 8, REPR::n, REPR::n);
    autodiff::VectorXreal XF8true{{0.8599481023526306, 1.2105053156859162, 1.7337079805154423, 2.2545824516802759,
                                   2.7454175483197241, 3.2662920194845584, 3.7894946843140835, 4.1400518976473686}};

    EXPECT_TRUE(XF4.isApprox(X));
    EXPECT_TRUE(XF8.isApprox(XF8true));
}

// Test multiply functions
TEST(SkellyChebyshev, multiply) {
    // Test the multiply functionality
    autodiff::VectorXreal X{{1, 2, 3, 4}};
    autodiff::VectorXreal Y{{5, 6, 7, 8}};

    autodiff::VectorXreal ZC = Multiply(X, Y, REPR::c, REPR::c, REPR::c);
    autodiff::VectorXreal ZF = Multiply(X, Y, REPR::n, REPR::n, REPR::n);
    autodiff::VectorXreal ZCtrue{{37.5000000000000071, 57.9999999999999929, 47.9999999999999929, 44.0000000000000000}};
    autodiff::VectorXreal ZFtrue{{5.0044417382415922, 11.9955582617584042, 20.9955582617584078, 32.0044417382415958}};

    EXPECT_TRUE(ZC.isApprox(ZCtrue));
    EXPECT_TRUE(ZF.isApprox(ZFtrue));
}

// Test evaluation of polynomials
TEST(SkellyChebyshev, evalpoly) {
    // Create Chebyshev coefficients we can use to test
    autodiff::VectorXreal initX{{0.1, 0.2, 0.3, 0.4}};
    autodiff::real mone{-1.0};
    autodiff::real xeval = evalpoly(mone, initX);
    EXPECT_DOUBLE_EQ(-0.2, xeval.val());

    // Check that we can do the left evaluation of the same
    autodiff::VectorXreal initY{{0.1, 0.2, 0.3, 0.4, 0.5}};
    autodiff::real yevalL = LeftEvalPoly<autodiff::real, autodiff::VectorXreal>(initY);
    autodiff::real yevalR = RightEvalPoly<autodiff::real, autodiff::VectorXreal>(initY);
    EXPECT_DOUBLE_EQ(0.3, yevalL.val());
    EXPECT_DOUBLE_EQ(1.5, yevalR.val());
}