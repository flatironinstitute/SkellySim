// @HEADER
// @HEADER

#ifndef FIBER_CHEBYSHEV_PENALTY_AUTODIFF_HPP_
#define FIBER_CHEBYSHEV_PENALTY_AUTODIFF_HPP_

/// \file fiber_chebyshev_penalty_autodiff.hpp
/// \brief Fibers via chebyshev penalty implementation (with integration)
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
#include <tuple>

// SkellySim libs
#include <fiber_state.hpp>
#include <skelly_chebyshev.hpp>
#include <skelly_fiber.hpp>
#include <skelly_sim.hpp>

namespace skelly_fiber {

template <typename VecT>
class FiberChebyshevPenaltyAutodiff {
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

    /// @brief Construct a fiber of a given discretizaiton
    FiberChebyshevPenaltyAutodiff(int n_nodes, int n_nodes_tension, int n_equations, int n_equations_tension)
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
    }

    // Split up the main equations into whatever we need
    std::tuple<VecT, VecT, VecT> SplitMain(const VecT &x) {
        VecT X = x.segment(0, n_nodes_);
        VecT Y = x.segment(n_nodes_, n_nodes_);
        VecT T = x.segment(2 * n_nodes_, n_nodes_tension_);

        return std::make_tuple(X, Y, T);
    }
    std::tuple<VecT, VecT> SplitAt(const VecT &x, unsigned int n) {
        VecT X1 = x.segment(0, n);
        VecT X2 = x.segment(n, x.size() - n);
        return std::make_tuple(X1, X2);
    }

    // XXX Make sure that the move semantics are correct for the FiberState object here!!!!!
    FiberState<VecT> DivideAndConstruct(const VecT &XX, const double L) {
        // Create a return object
        // XXX Check the number of dimensions later on this, for 3d, as this does the init of the size of things to make
        // sure that we can tie to them later...
        FiberState<VecT> Div(2, n_nodes_, n_nodes_tension_, n_equations_, n_equations_tension_);
        Div.XX_ = XX;
        // Divide up the main state array into its components
        auto [XW, YW, TW] = SplitMain(XX);

        VecT Dx;
        VecT Dy;
        VecT Dt;
        std::tie(Div.XssssC_, Dx) = SplitAt(XW, n_equations_);
        std::tie(Div.YssssC_, Dy) = SplitAt(YW, n_equations_);
        std::tie(Div.TssC_, Dt) = SplitAt(TW, n_equations_tension_);

        // Now integrate up our solution
        double rat = L / 2.0;
        std::tie(Div.XsssC_, Div.XssC_, Div.XsC_, Div.XC_) = IntegrateUp4(Div.XssssC_, rat, Dx);
        std::tie(Div.YsssC_, Div.YssC_, Div.YsC_, Div.YC_) = IntegrateUp4(Div.YssssC_, rat, Dy);
        std::tie(Div.TsC_, Div.TC_) = IntegrateUpTension2(Div.TssC_, rat, Dt);

        return Div;
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
    // Tension 2nd -> 1st derivative
    std::tuple<VecT, VecT> IntegrateUpTension2(const VecT &TssC, const double rat, const VecT &BT) {
        VecT TsC = (IMT_ * TssC) * rat;
        VecT TC;
        TsC[0] += BT[BT.size() - 1];
        VecT AT = BT.segment(0, BT.size() - 1);
        std::tie(TC) = IntegrateUpTension1(TsC, rat, AT);
        return std::make_tuple(TsC, TC);
    }
    // Tension 1st -> 0th derivative
    std::tuple<VecT> IntegrateUpTension1(const VecT &TsC, const double rat, const VecT &AT) {
        VecT TC = (IMT_ * TsC) * rat;
        TC[0] += AT[AT.size() - 1];
        return std::make_tuple(TC);
    }

    // Write to console
    friend auto operator<<(std::ostream &os, const FiberChebyshevPenaltyAutodiff<VecT> &m) -> std::ostream & {
        os << "FiberSolver:\n";
        os << "...tension type:         Penalty\n";
        os << "...N / NT / Neq / NeqT:  " << m.n_nodes_ << ", " << m.n_nodes_tension_ << ", " << m.n_equations_ << ", "
           << m.n_equations_tension_ << std::endl;
        return os;
    }
};

/// @brief Fiber penalty deflection objective (TEST)
template <typename VecT>
VecT SheerDeflectionObjective(const VecT &XX, FiberChebyshevPenaltyAutodiff<VecT> &FS, const VecT &oldXX,
                              const double L, const double zeta, const double dt) {
    // Set up a cute 'using' to get the internal scalar type of VecT
    using ScalarT = typename VecT::Scalar;

    auto Div = FS.DivideAndConstruct(XX, L);
    auto oDiv = FS.DivideAndConstruct(oldXX, L);

    // Get the forces
    auto [FxC, FyC, AFxC, AFyC] = FiberForces(Div, oDiv, 1.0, FS.n_equations_);

    // Sheer velocity
    VecT UC = zeta * Div.YC_;
    VecT VC = VecT::Zero(Div.YC_.size());
    VecT UsC = zeta * Div.YsC_;
    VecT VsC = VecT::Zero(Div.YsC_.size());
    VecT oUsC = zeta * oDiv.YsC_;
    VecT oVsC = VecT::Zero(oDiv.YsC_.size());

    // Get the evolution equations
    auto [teqXC, teqYC] = FiberEvolution(AFxC, AFyC, Div, oDiv, UC, VC, dt);

    // Get the tension equation
    auto teqTC = FiberPenaltyTension(Div, oDiv, UsC, VsC, oUsC, oVsC, dt, FS.n_equations_tension_);

    // Get the boundary conditions
    VecT cposition{{0.0, 0.0}};
    VecT cdirector{{0.0, 1.0}};
    FiberBoundaryCondition<ScalarT> BCL = ClampedBC<ScalarT, VecT>(Div, oDiv, FSIDE::left, cposition, cdirector);
    FiberBoundaryCondition<ScalarT> BCR = FreeBC<ScalarT, VecT>(Div, FSIDE::right);

    // Combine together boundary conditions with the equations
    auto eqXC = CombineXWithBCs<ScalarT, VecT>(teqXC, BCL, BCR);
    auto eqYC = CombineYWithBCs<ScalarT, VecT>(teqYC, BCL, BCR);
    auto eqTC = CombineTPenaltyWithBCs<ScalarT, VecT>(teqTC, BCL, BCR);

    VecT eq_full(eqXC.size() + eqYC.size() + eqTC.size());
    eq_full << eqXC, eqYC, eqTC;
    return eq_full;
}

} // namespace skelly_fiber

#endif // FIBER_CHEBYSHEV_PENALTY_AUTODIFF_HPP_
