
/// \file unit_test_fiber_base.cpp
/// \brief Unit tests for FiberBase class

// C++ includes
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// skelly includes
#include <fiber_base.hpp>
#include <msgpack.hpp>

// test files
#include "./mpi_environment.hpp"

// Test if the Chebyshev convenience functions work correctly
TEST(FiberBase, constructor) {
    int N = 16;
    FiberBase FS(N, N - 3, N - 4, N - 4);

    Eigen::MatrixXd FS_IM_true{{0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                                0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                                0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
                               {1.0000000000000000, 0.0000000000000000, -0.5000000000000000, 0.0000000000000000,
                                0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                                0.0000000000000000, 0.0000000000000000, -0.0000000000000000, 0.0000000000000000},
                               {0.0000000000000000, 0.2500000000000000, 0.0000000000000000, -0.2500000000000000,
                                0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                                0.0000000000000000, 0.0000000000000000, -0.0000000000000000, 0.0000000000000000},
                               {0.0000000000000000, 0.0000000000000000, 0.1666666666666667, 0.0000000000000000,
                                -0.1666666666666667, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                                -0.0000000000000000, -0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
                               {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.1250000000000000,
                                0.0000000000000000, -0.1250000000000000, 0.0000000000000000, 0.0000000000000000,
                                0.0000000000000000, 0.0000000000000000, -0.0000000000000000, 0.0000000000000000},
                               {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                                0.1000000000000000, 0.0000000000000000, -0.1000000000000000, 0.0000000000000000,
                                0.0000000000000000, 0.0000000000000000, -0.0000000000000000, 0.0000000000000000},
                               {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                                0.0000000000000000, 0.0833333333333333, 0.0000000000000000, -0.0833333333333333,
                                0.0000000000000000, 0.0000000000000000, -0.0000000000000000, 0.0000000000000000},
                               {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                                0.0000000000000000, 0.0000000000000000, 0.0714285714285714, 0.0000000000000000,
                                -0.0714285714285714, 0.0000000000000000, -0.0000000000000000, 0.0000000000000000},
                               {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                                0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0625000000000000,
                                0.0000000000000000, -0.0625000000000000, -0.0000000000000000, 0.0000000000000000},
                               {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                                0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                                0.0555555555555556, 0.0000000000000000, -0.0555555555555556, 0.0000000000000000},
                               {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                                0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                                0.0000000000000000, 0.0500000000000000, -0.0000000000000000, 0.0000000000000000},
                               {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                                0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                                0.0000000000000000, 0.0000000000000000, 0.0454545454545455, 0.0000000000000000}};
    EXPECT_TRUE(FS.IM_.isApprox(FS_IM_true));
}

// Test our ability to do 'views' into the object
TEST(FiberBase, views) {
    // Create a smaller fiber
    int N = 8;
    FiberBase FS(N, N - 3, N - 4, N - 4);

    // Create a canned state vector
    Eigen::VectorXd initXX{
        {1, 2, 3, 4, 5, 6, 7, 8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.01, 0.02, 0.03, 0.04, 0.05}};
    FS.XX_ = initXX;
    EXPECT_TRUE(initXX.segment(0, 8).isApprox(FS.XW()));
    EXPECT_TRUE(initXX.segment(8, 8).isApprox(FS.YW()));
    EXPECT_TRUE(initXX.segment(16, 5).isApprox(FS.TW()));

    // Try views into the split_at function
    auto s1 = FS.SplitX1(FS.XW(), 4);
    auto s2 = FS.SplitX2(FS.XW(), 4);
    EXPECT_TRUE(s1.isApprox(initXX.segment(0, 4)));
    EXPECT_TRUE(s2.isApprox(initXX.segment(4, 4)));

    // Can we edit into the double-ref?
    s1[3] = 599;
    Eigen::VectorXd s1_new{{1, 2, 3, 599}};
    EXPECT_TRUE(s1_new.isApprox(s1));
}

// Test divide and construct abilities
TEST(FiberBase, divide_and_construct) {
    int N = 8;
    FiberBase FS(N, N - 3, N - 4, N - 4);

    // Create a canned state vector
    Eigen::VectorXd initXX{
        {1, 2, 3, 4, 5, 6, 7, 8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.01, 0.02, 0.03, 0.04, 0.05}};
    FS.XX_ = initXX;

    // Call divide and construct, even though some of the values are bogus, to see if they are constructed internally
    // correctly
    FS.DivideAndConstruct(0.5);
    EXPECT_TRUE(FS.XssssC_.isApprox(initXX.segment(0, 4)));
    EXPECT_TRUE(FS.YssssC_.isApprox(initXX.segment(8, 4)));
    EXPECT_TRUE(FS.TsC_.isApprox(initXX.segment(16, 4)));

    // Check the integrations in XC
    Eigen::VectorXd XsssC_true{{48.0, -0.125, 0.125, 0.125}};
    Eigen::VectorXd XssC_true{{14.0, 11.984375, -0.0078125, 0.005208333333333333}};
    Eigen::VectorXd XsC_true{{6.0, 3.5009765625, 0.7490234375, -0.0003255208333333333}};
    Eigen::VectorXd XC_true{{5.0, 1.4063720703125, 0.21881103515625, 0.031209309895833332}};

    EXPECT_TRUE(XsssC_true.isApprox(FS.XsssC_));
    EXPECT_TRUE(XssC_true.isApprox(FS.XssC_));
    EXPECT_TRUE(XsC_true.isApprox(FS.XsC_));
    EXPECT_TRUE(XC_true.isApprox(FS.XC_));

    // Check the integrations in T
    Eigen::VectorXd TC_true{{0.05, -0.00125, 0.00125, 0.00125}};
    EXPECT_TRUE(TC_true.isApprox(FS.TC_));
}

// Test views into the arrays of the fiber (divide and construct)
TEST(FiberBase, integration) {
    // Create a fiber base object to store data
    int N = 16;
    FiberBase FS(N, N - 3, N - 4, N - 4);
    double L = 0.1;

    // initialization conditions taken from julia to compare
    Eigen::VectorXd init_X{{0.9997588300226782, 0.9058773132192033, 0.6972363395327937, 0.05311960018683659,
                            0.9073058495439762, 0.5844709323969469, 0.9926879198264066, 0.8174088784626208,
                            0.6226889681098617, 0.944194248057808, 0.5264407345576302, 0.14594985883286538,
                            0.4044355011201404, 0.7273031813753761, 0.1779453022879729, 0.8886702751074465}};
    Eigen::VectorXd init_Y{{0.0717673027334802, 0.4141150124525852, 0.1293327638944125, 0.41547171378753767,
                            0.25002945685031575, 0.8789343034881281, 0.10127226406163892, 0.5856798770901475,
                            0.8913856789767505, 0.8582499568770857, 0.997423121916566, 0.8218839755299836,
                            0.04553314891254823, 0.4026486459151246, 0.8599775977324692, 0.7524155535683011}};
    Eigen::VectorXd init_T{{0.18311605785772567, 0.17679324432680055, 0.7696932851561678, 0.013768311103934616,
                            0.9854863423547627, 0.1300696339820414, 0.4064240831808379, 0.13709530013691218,
                            0.8522195831731575, 0.10551693315332411, 0.35052090551564186, 0.6938329801718334,
                            0.5987492088519557}};

    // concatenate vectors
    Eigen::VectorXd XX(init_X.size() + init_Y.size() + init_T.size());
    XX << init_X, init_Y, init_T;
    FS.XX_ = XX;

    // Run divide and construct
    FS.DivideAndConstruct(L);

    // Test the tension component
    auto IMT = FS.IMT_ * (L / 2.0);
    Eigen::MatrixXd TsM(FS.n_equations_tension_, FS.n_equations_tension_ + 1);
    TsM << Eigen::MatrixXd::Identity(FS.n_equations_tension_, FS.n_equations_tension_),
        Eigen::MatrixXd::Zero(FS.n_equations_tension_, 1);
    Eigen::VectorXd colvec(FS.n_equations_tension_);
    colvec << 1.0, Eigen::VectorXd::Zero(FS.n_equations_tension_ - 1);
    Eigen::MatrixXd TM(IMT.rows(), IMT.cols() + 1);
    TM << IMT, colvec;
    auto TsC_raw = TsM * init_T;
    auto TC_raw = TM * init_T;

    EXPECT_TRUE(TC_raw.isApprox(FS.TC_));
    EXPECT_TRUE(TsC_raw.isApprox(FS.TsC_));
}
