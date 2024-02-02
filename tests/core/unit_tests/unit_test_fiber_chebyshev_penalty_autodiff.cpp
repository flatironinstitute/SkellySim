
/// \file unit_test_fiber_base.cpp
/// \brief Unit tests for FiberChebyshevPenalty class

// C++ includes
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// skelly includes
#include <fiber_chebyshev_penalty_autodiff.hpp>
#include <msgpack.hpp>

// test files
#include "./mpi_environment.hpp"

// Test the penalty version constructor
TEST(FiberChebysehvPenaltyAutodiff, real_constructor) {
    // Create a fiber object
    int N = 20;
    int NT = N - 2;
    int Neq = N - 4;
    int NTeq = NT - 2;
    double mlength = 1.0;
    FiberChebyshevPenaltyAutodiff<autodiff::VectorXreal> FS(N, NT, Neq, NTeq);
    std::cout << FS;

    // Now create our fiber in XYT
    Eigen::VectorXd init_X = Eigen::VectorXd::Zero(FS.n_nodes_);
    Eigen::VectorXd init_Y = Eigen::VectorXd::Zero(FS.n_nodes_);
    Eigen::VectorXd init_T = Eigen::VectorXd::Zero(FS.n_nodes_tension_);
    init_Y[init_Y.size() - 1 - 3] = mlength / 2.0;
    init_Y[init_Y.size() - 1 - 2] = 1.0;
    Eigen::VectorXd XX(init_X.size() + init_Y.size() + init_T.size());
    XX << init_X, init_Y, init_T;
    std::cout << "Initial vector:\n" << XX << "\n    size: " << XX.size() << std::endl;

    // Try Divide and Construct
    auto Div = FS.DivideAndConstruct(XX, mlength);
    std::cout << "Div:\n" << Div << std::endl;

    autodiff::VectorXreal YCtrue{{0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    autodiff::VectorXreal YsCtrue{{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    EXPECT_TRUE(YCtrue.isApprox(Div.YC_));
    EXPECT_TRUE(YsCtrue.isApprox(Div.YsC_));
}

// Test the penalty forces
TEST(FiberChebysehvPenaltyAutodiff, real_forces) {
    // Create a fiber object
    int N = 20;
    int NT = N - 2;
    int Neq = N - 4;
    int NTeq = NT - 2;
    double mlength = 1.0;
    FiberChebyshevPenaltyAutodiff<autodiff::VectorXreal> FS(N, NT, Neq, NTeq);
    std::cout << FS;

    // Now create our fiber in XYT
    Eigen::VectorXd init_X = Eigen::VectorXd::Zero(FS.n_nodes_);
    Eigen::VectorXd init_Y = Eigen::VectorXd::Zero(FS.n_nodes_);
    Eigen::VectorXd init_T = Eigen::VectorXd::Zero(FS.n_nodes_tension_);
    init_Y[init_Y.size() - 1 - 3] = mlength / 2.0;
    init_Y[init_Y.size() - 1 - 2] = 1.0;
    Eigen::VectorXd XX(init_X.size() + init_Y.size() + init_T.size());
    Eigen::VectorXd oldXX(init_X.size() + init_Y.size() + init_T.size());
    XX << init_X, init_Y, init_T;
    oldXX = XX;

    // Create the Div and oDiv structures for the forces
    auto Div = FS.DivideAndConstruct(XX, mlength);
    auto oDiv = FS.DivideAndConstruct(oldXX, mlength);

    std::cout << "Div construction:\n";
    std::cout << "Div: \n" << Div << std::endl;
    std::cout << "oDiv:\n" << oDiv << std::endl;
}

// ********************************************************************************************************************
// Physics tests
// See if we can do a full physics test
// ********************************************************************************************************************

// // Test a full sheer flow implementation
// TEST(FiberChebysehvPenaltyAutodiff, construction) {
//     // Create a fiber object
//     int N = 20;
//     int NT = N - 3;
//     int Neq = N - 4;
//     int NTeq = NT - 1;
//     double mlength = 1.0;
//     FiberChebyshevConstraintAutodiff<autodiff::VectorXreal> FS(N, NT, Neq, NTeq);
//     std::cout << FS;

//     // Now create our fiber in XYT
//     Eigen::VectorXd init_X = Eigen::VectorXd::Zero(FS.n_nodes_);
//     Eigen::VectorXd init_Y = Eigen::VectorXd::Zero(FS.n_nodes_);
//     Eigen::VectorXd init_T = Eigen::VectorXd::Zero(FS.n_nodes_tension_);
//     init_Y[init_Y.size() - 1 - 3] = mlength/2.0;
//     init_Y[init_Y.size() - 1 - 2] = 1.0;
//     Eigen::VectorXd XX(init_X.size() + init_Y.size() + init_T.size());
//     XX << init_X, init_Y, init_T;
//     std::cout << "Initial vector:\n" << XX << "\n    size: " << XX.size() << std::endl;
// }