// @HEADER
// @HEADER

#ifndef FIBER_CHEBYSHEV_PENALTY_AUTODIFF_EXTERNAL_HPP_
#define FIBER_CHEBYSHEV_PENALTY_AUTODIFF_EXTERNAL_HPP_

/// \file fiber_chebyshev_penalty_autodiff_external.hpp
/// \brief Fibers via chebyshev penalty implementation (with integration)
///
/// New Fiber base class to implement all of the different Chebyshev/spectral fibers. External version where we call
/// things like functions acting on states.

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
class FiberChebyshevPenaltyAutodiffExternal {
  public:
    // typedefs
    typedef VecT vector_type;

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

    /// @brief Default constructor
    FiberChebyshevPenaltyAutodiffExternal() = default;

    /// @brief Construct a fiber of a given discretizaiton
    FiberChebyshevPenaltyAutodiffExternal(int n_nodes, int n_nodes_tension, int n_equations, int n_equations_tension)
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

    // Write to console
    friend auto operator<<(std::ostream &os, const FiberChebyshevPenaltyAutodiffExternal<VecT> &m) -> std::ostream & {
        os << "FiberSolver:\n";
        os << "...tension type:         Penalty\n";
        os << "...N / NT / Neq / NeqT:  " << m.n_nodes_ << ", " << m.n_nodes_tension_ << ", " << m.n_equations_ << ", "
           << m.n_equations_tension_;
        return os;
    }

    // Some getters of internal variables
    // Get the local node count
    int get_local_node_count() const { return n_nodes_; }

    // Get the local solution size
    int get_local_solution_size() const { return 2 * n_nodes_ + n_nodes_tension_; }
};

/// @brief Fiber penalty deflection objective (TEST)
//
// XXX Should the FiberSolver be const?
template <typename VecT>
VecT SheerDeflectionObjectiveExternal(const VecT &XX, FiberChebyshevPenaltyAutodiffExternal<VecT> &FS,
                                      const VecT &oldXX, const double L, const double zeta, const double dt) {
    // Set up a cute 'using' to get the internal scalar type of VecT
    using ScalarT = typename VecT::Scalar;

    auto Div = DivideAndConstructFCPA(XX, L, FS.n_nodes_, FS.n_nodes_tension_, FS.n_equations_, FS.n_equations_tension_,
                                      FS.IM_, FS.IMT_);
    auto oDiv = DivideAndConstructFCPA(oldXX, L, FS.n_nodes_tension_, FS.n_equations_, FS.n_equations_tension_, FS.IM_,
                                       FS.IMT_);

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

/// @brief Helper function to construct a FiberSolver of this type and an initial state vector
///
/// Note: This is specific to an initial condition and the penalty autodiff types
template <typename VecT>
std::tuple<FiberChebyshevPenaltyAutodiffExternal<VecT>, VecT> SetupSolverInitialstateFCPA(const int N, const double L) {
    // Set up the numbers of things that are going on
    int NT = N - 2;
    int Neq = N - 4;
    int NTeq = NT - 2;

    // Create the fiber solver
    FiberChebyshevPenaltyAutodiffExternal<VecT> FS = FiberChebyshevPenaltyAutodiffExternal<VecT>(N, NT, Neq, NTeq);

    // Now create our fiber in XYT
    VecT init_X = Eigen::VectorXd::Zero(FS.n_nodes_);
    VecT init_Y = Eigen::VectorXd::Zero(FS.n_nodes_);
    VecT init_T = Eigen::VectorXd::Zero(FS.n_nodes_tension_);
    init_Y[init_Y.size() - 1 - 3] = L / 2.0;
    init_Y[init_Y.size() - 1 - 2] = 1.0;
    VecT XX(init_X.size() + init_Y.size() + init_T.size());
    XX << init_X, init_Y, init_T;

    // return a tuple of these
    return std::make_tuple(FS, XX);
}

/// @brief Extricate the 'XC' like variables and the extensibility error
template <typename T, typename VecT>
std::tuple<VecT, VecT, VecT, T> Extricate(const VecT &XX, FiberChebyshevPenaltyAutodiffExternal<VecT> &FS,
                                          const double L) {
    // Get the components of XX
    auto Div = DivideAndConstructFCPA(XX, L);

    auto ext_error = skelly_fiber::ExtensibilityError<T, VecT>(Div.XsC_, Div.YsC_);
    return std::make_tuple(Div.XC_, Div.YC_, Div.TC_, ext_error);
}

////////
// New functional form of divide and construct and integrate functions
////////

/// @brief Functional form of SplitMain
template <typename VecT>
inline std::tuple<VecT, VecT, VecT> SplitMain(const VecT &x, const int n_nodes, const int n_nodes_tension) {
    return std::make_tuple(x.segment(0, n_nodes), x.segment(n_nodes, n_nodes), x.segment(2 * n_nodes, n_nodes_tension));
}

/// @brief Functional form of SplitAt
template <typename VecT>
inline std::tuple<VecT, VecT> SplitAt(const VecT &x, const unsigned int n) {
    return std::make_tuple(x.segment(0, n), x.segment(n, x.size() - n));
}

// Do all 4 derivatives for fibers, return as tuples
// 1st -> 0th derivative
template <typename VecT>
inline std::tuple<VecT> IntegrateUp1(const VecT &XsC, const double rat, const VecT &AX, CMatrixRef &IM) {
    VecT XC = (IM * XsC) * rat;
    XC[0] += AX[AX.size() - 1];
    return std::make_tuple(XC);
}
// 2nd -> 1st derivative
template <typename VecT>
inline std::tuple<VecT, VecT> IntegrateUp2(const VecT &XssC, const double rat, const VecT &BX, CMatrixRef &IM) {
    VecT XsC = (IM * XssC) * rat;
    VecT XC;
    XsC[0] += BX[BX.size() - 1];
    VecT AX = BX.segment(0, BX.size() - 1);
    std::tie(XC) = IntegrateUp1(XsC, rat, AX, IM);
    return std::make_tuple(XsC, XC);
}
// 3rd -> 2nd derivative
template <typename VecT>
inline std::tuple<VecT, VecT, VecT> IntegrateUp3(const VecT &XsssC, const double rat, const VecT &CX, CMatrixRef &IM) {
    VecT XssC = (IM * XsssC) * rat;
    VecT XsC;
    VecT XC;
    XssC[0] += 2.0 * CX[CX.size() - 1];
    VecT BX = CX.segment(0, CX.size() - 1);
    std::tie(XsC, XC) = IntegrateUp2(XssC, rat, BX, IM);
    return std::make_tuple(XssC, XsC, XC);
}
// 4th -> 3rd derivative
template <typename VecT>
inline std::tuple<VecT, VecT, VecT, VecT> IntegrateUp4(const VecT &XssssC, const double rat, const VecT &DX,
                                                       CMatrixRef &IM) {
    VecT XsssC = (IM * XssssC) * rat;
    VecT XssC;
    VecT XsC;
    VecT XC;
    XsssC[0] += 6.0 * DX[DX.size() - 1];
    VecT CX = DX.segment(0, DX.size() - 1);
    std::tie(XssC, XsC, XC) = IntegrateUp3(XsssC, rat, CX, IM);
    return std::make_tuple(XsssC, XssC, XsC, XC);
}

// Tension 1st -> 0th derivative
template <typename VecT>
inline std::tuple<VecT> IntegrateUpTension1(const VecT &TsC, const double rat, const VecT &AT, CMatrixRef &IMT) {
    VecT TC = (IMT * TsC) * rat;
    TC[0] += AT[AT.size() - 1];
    return std::make_tuple(TC);
}
// Tension 2nd -> 1st derivative
template <typename VecT>
inline std::tuple<VecT, VecT> IntegrateUpTension2(const VecT &TssC, const double rat, const VecT &BT, CMatrixRef &IMT) {
    VecT TsC = (IMT * TssC) * rat;
    VecT TC;
    TsC[0] += BT[BT.size() - 1];
    VecT AT = BT.segment(0, BT.size() - 1);
    std::tie(TC) = IntegrateUpTension1(TsC, rat, AT, IMT);
    return std::make_tuple(TsC, TC);
}

// XXX Make sure that the move semantics are correct for the FiberState object here!!!!!
template <typename VecT>
inline FiberState<VecT> DivideAndConstructFCPA(const VecT &XX, const double L, const int n_nodes,
                                               const int n_nodes_tension, const int n_equations,
                                               const int n_equations_tension, CMatrixRef &IM, CMatrixRef &IMT) {
    // Create a return object
    // XXX Check the number of dimensions later on this, for 3d, as this does the init of the size of things to make
    // sure that we can tie to them later...
    FiberState<VecT> Div(2, n_nodes, n_nodes_tension, n_equations, n_equations_tension);
    Div.XX_ = XX;
    // Divide up the main state array into its components
    auto [XW, YW, TW] = SplitMain(XX, n_nodes, n_nodes_tension);

    VecT Dx;
    VecT Dy;
    VecT Dt;
    std::tie(Div.XssssC_, Dx) = SplitAt(XW, n_equations);
    std::tie(Div.YssssC_, Dy) = SplitAt(YW, n_equations);
    std::tie(Div.TssC_, Dt) = SplitAt(TW, n_equations_tension);

    // Now integrate up our solution
    double rat = L / 2.0;
    std::tie(Div.XsssC_, Div.XssC_, Div.XsC_, Div.XC_) = IntegrateUp4(Div.XssssC_, rat, Dx, IM);
    std::tie(Div.YsssC_, Div.YssC_, Div.YsC_, Div.YC_) = IntegrateUp4(Div.YssssC_, rat, Dy, IM);
    std::tie(Div.TsC_, Div.TC_) = IntegrateUpTension2(Div.TssC_, rat, Dt, IMT);

    return Div;
}

} // namespace skelly_fiber

#endif // FIBER_CHEBYSHEV_PENALTY_AUTODIFF_EXTERNAL_HPP_
