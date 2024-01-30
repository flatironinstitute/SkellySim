// @HEADER
// @HEADER

#ifndef FIBER_AUTODIFF_HPP_
#define FIBER_AUTODIFF_HPP_

/// \file fiber_chebyshev_constraint.hpp
/// \brief Fibers via chebyshev constraint implementation (with integration)
///
/// New Fiber base class to implement all of the different Chebyshev/spectral fibers

// External libs
#include <Eigen/LU>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

// C libs
#include <math.h>

// C++ core libs
#include <iostream>

// SkellySim libs
#include <skelly_chebyshev.hpp>
#include <skelly_sim.hpp>

template <typename VecT>
class FiberChebyshevConstraintAutodiff {
  public:
    // Input parameters
    unsigned int n_nodes_;             ///< number of nodes representing XYZ
    unsigned int n_nodes_tension_;     ///< number of nodes representing tension
    unsigned int n_equations_;         ///< number of equations to use for XYZ
    unsigned int n_equations_tension_; ///< number of equations to use for tension

    // Node variables
    Eigen::VectorXd s_;
    Eigen::VectorXd sT_;
    Eigen::VectorXd sNeq_;
    Eigen::VectorXd sNeqT_;

    // Integration matrices
    // XXX Might be moved into a subclass of this later, for now, code up here
    Eigen::MatrixXd IM_;
    Eigen::MatrixXd IMT_;

    // State vectors
    //
    // Include all of the derivatives, as we usually want to keep them around
    VecT XX_;     ///< concatenated entire state of system
    VecT XssssC_; ///< Fourth derivative of X components in coefficient space
    VecT XsssC_;  ///< Third derivative of X components in coefficient space
    VecT XssC_;   ///< Second derivative of X components in coefficient space
    VecT XsC_;    ///< First derivative of X components in coefficient space
    VecT XC_;     ///< Zeroth derivative of X components in coefficient space
    VecT YssssC_; ///< Fourth derivative of Y components in coefficient space
    VecT YsssC_;  ///< Third derivative of Y components in coefficient space
    VecT YssC_;   ///< Second derivative of Y components in coefficient space
    VecT YsC_;    ///< First derivative of Y components in coefficient space
    VecT YC_;     ///< Zeroth derivative of Y components in coefficient space
    VecT TssC_;   ///< Second derivative of T components in coefficient space
    VecT TsC_;    ///< First derivative of T components in coefficient space
    VecT TC_;     ///< Zeroth derivative of T components in coefficient space

    /// @brief Construct a fiber of a given discretizaiton
    FiberChebyshevConstraintAutodiff(int n_nodes, int n_nodes_tension, int n_equations, int n_equations_tension)
        : n_nodes_(n_nodes), n_nodes_tension_(n_nodes_tension), n_equations_(n_equations),
          n_equations_tension_(n_equations_tension) {
        // Generate all of the Chebyshev information
        s_ = skelly_chebyshev::ChebyshevTPoints(0.0, 1.0, n_nodes_);
        sT_ = skelly_chebyshev::ChebyshevTPoints(0.0, 1.0, n_nodes_tension_);
        sNeq_ = skelly_chebyshev::ChebyshevTPoints(0.0, 1.0, n_equations_);
        sNeqT_ = skelly_chebyshev::ChebyshevTPoints(0.0, 1.0, n_equations_tension_);

        // Create the two integration matrices
        IM_ = skelly_chebyshev::IntegrationMatrix(n_equations_, skelly_chebyshev::REPR::c, skelly_chebyshev::REPR::c);
        IMT_ = skelly_chebyshev::IntegrationMatrix(n_equations_tension_, skelly_chebyshev::REPR::c,
                                                   skelly_chebyshev::REPR::c);
        IM_.row(0).setZero();
        IMT_.row(0).setZero();

        // Set the main state vectors to their proper size and zero out. Have to create them explicitly, otherwise,
        // later VectorRef operations will cause segfaults
        XX_ = Eigen::VectorXd::Zero(2 * n_nodes_ + n_nodes_tension_);
        XssssC_ = Eigen::VectorXd::Zero(n_equations_);
        XsssC_ = Eigen::VectorXd::Zero(n_equations_);
        XssC_ = Eigen::VectorXd::Zero(n_equations_);
        XsC_ = Eigen::VectorXd::Zero(n_equations_);
        XC_ = Eigen::VectorXd::Zero(n_equations_);
        YssssC_ = Eigen::VectorXd::Zero(n_equations_);
        YsssC_ = Eigen::VectorXd::Zero(n_equations_);
        YssC_ = Eigen::VectorXd::Zero(n_equations_);
        YsC_ = Eigen::VectorXd::Zero(n_equations_);
        YC_ = Eigen::VectorXd::Zero(n_equations_);
        TssC_ = Eigen::VectorXd::Zero(n_equations_);
        TsC_ = Eigen::VectorXd::Zero(n_equations_);
        TC_ = Eigen::VectorXd::Zero(n_equations_);
    }

    // Split up the main equations into whatever we need
    std::tuple<VecT, VecT, VecT> SplitMain() {
        VecT X = XX_.segment(0, n_nodes_);
        VecT Y = XX_.segment(n_nodes_, n_nodes_);
        VecT T = XX_.segment(2 * n_nodes_, n_nodes_tension_);

        return std::make_tuple(X, Y, T);
    }
    std::tuple<VecT, VecT> SplitAt(const VecT &x, unsigned int n) {
        VecT X1 = x.segment(0, n);
        VecT X2 = x.segment(n, x.size() - n);
        return std::make_tuple(X1, X2);
    }

    // Divide and consturct everything like the non-autodiff version
    void DivideAndConstruct(double L) {
        // Divide up the main state array into its components
        auto [XW, YW, TW] = SplitMain();

        VecT Dx;
        VecT Dy;
        VecT Dt;
        std::tie(XssssC_, Dx) = SplitAt(XW, n_equations_);
        std::tie(YssssC_, Dy) = SplitAt(YW, n_equations_);
        std::tie(TsC_, Dt) = SplitAt(TW, n_equations_tension_);

        // Now integrate up our solution
        double rat = L / 2.0;
        std::tie(XsssC_, XssC_, XsC_, XC_) = IntegrateUp4(XssssC_, rat, Dx);
        std::tie(YsssC_, YssC_, YsC_, YC_) = IntegrateUp4(YssssC_, rat, Dy);
        std::tie(TC_) = IntegrateUpTension1(TsC_, rat, Dt);
    }

    // Do all 4 derivatives for fibers, return as tuples
    // 4th -> 3rd derivative
    std::tuple<VecT, VecT, VecT, VecT> IntegrateUp4(const VecT &XssssC, const double rat, const VecT &DX) {
        VecT XsssC = (IM_ * XssssC) * rat;
        VecT XssC;
        VecT XsC;
        VecT XC;
        XsssC[0] += 6.0 * DX[DX.size() - 1];
        VecT CX = DX.segment(0, DX.size() - 1);
        std::tie(XssC, XsC, XC) = IntegrateUp3(XsssC, rat, CX);
        return std::make_tuple(XsssC, XssC, XsC, XC);
    }
    // 3rd -> 2nd derivative
    std::tuple<VecT, VecT, VecT> IntegrateUp3(const VecT &XsssC, const double rat, const VecT &CX) {
        VecT XssC = (IM_ * XsssC) * rat;
        VecT XsC;
        VecT XC;
        XssC[0] += 2.0 * CX[CX.size() - 1];
        VecT BX = CX.segment(0, CX.size() - 1);
        std::tie(XsC, XC) = IntegrateUp2(XssC, rat, BX);
        return std::make_tuple(XssC, XsC, XC);
    }
    // 2nd -> 1st derivative
    std::tuple<VecT, VecT> IntegrateUp2(const VecT &XssC, const double rat, const VecT &BX) {
        VecT XsC = (IM_ * XssC) * rat;
        VecT XC;
        XsC[0] += BX[BX.size() - 1];
        VecT AX = BX.segment(0, BX.size() - 1);
        std::tie(XC) = IntegrateUp1(XsC, rat, AX);
        return std::make_tuple(XsC, XC);
    }
    // 1st -> 0th derivative
    std::tuple<VecT> IntegrateUp1(const VecT &XsC, const double rat, const VecT &AX) {
        VecT XC = (IM_ * XsC) * rat;
        XC[0] += AX[AX.size() - 1];
        return std::make_tuple(XC);
    }
    // Tension 1st -> 0th derivative
    std::tuple<VecT> IntegrateUpTension1(const VecT &TsC, const double rat, const VecT &AT) {
        VecT TC = (IMT_ * TsC) * rat;
        TC[0] += AT[AT.size() - 1];
        return std::make_tuple(TC);
    }
};

#endif // FIBER_BASE_HPP_
