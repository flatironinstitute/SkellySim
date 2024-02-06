// @HEADER
// @HEADER

#ifndef SKELLY_FIBER_HPP_
#define SKELLY_FIBER_HPP_

/// \file skelly_fiber.hpp
/// \brief Templated helper functions for (new) fiber implmentations
///
/// General functions for fibers common to all

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

namespace skelly_fiber {

/// @brief Enum for defining left/right side of fiber
enum FSIDE { left, right };

/// @brief Euler-Bernoulli beam forces with slender body theory
template <typename VecT>
std::tuple<VecT, VecT, VecT, VecT> FiberForces(const FiberState<VecT> &Div, const FiberState<VecT> &oDiv,
                                               const double E, const unsigned int n_equations) {
    // Grab the proper namespaces for things like multiply, REPR
    using skelly_chebyshev::Multiply;
    using skelly_chebyshev::REPR;

    // Alias our variables to make reading easier
    const VecT &XssssC = Div.XssssC_;
    const VecT &YssssC = Div.YssssC_;

    const VecT &TsC = Div.TsC_;
    const VecT &TC = Div.TC_;

    const VecT &nXssC = oDiv.XssC_;
    const VecT &nXsC = oDiv.XsC_;
    const VecT &nYssC = oDiv.YssC_;
    const VecT &nYsC = oDiv.YsC_;

    // Force densities
    VecT FxC = -E * XssssC + Multiply(TC, nXssC, REPR::c, REPR::c, REPR::c, n_equations) +
               Multiply(TsC, nXsC, REPR::c, REPR::c, REPR::c, n_equations);
    VecT FyC = -E * YssssC + Multiply(TC, nYssC, REPR::c, REPR::c, REPR::c, n_equations) +
               Multiply(TsC, nYsC, REPR::c, REPR::c, REPR::c, n_equations);
    // Compute slender body portion to get AF
    VecT AxxF = VecT::Ones(n_equations) + Multiply(nXsC, nXsC, REPR::c, REPR::c, REPR::n, n_equations);
    VecT AxyF = Multiply(nXsC, nYsC, REPR::c, REPR::c, REPR::n, n_equations);
    VecT AyyF = VecT::Ones(n_equations) + Multiply(nYsC, nYsC, REPR::c, REPR::c, REPR::n, n_equations);
    // Combine to get AFC (in coefficient space)
    VecT AFxC = Multiply(AxxF, FxC, REPR::n, REPR::c, REPR::c, n_equations) +
                Multiply(AxyF, FyC, REPR::n, REPR::c, REPR::c, n_equations);
    VecT AFyC = Multiply(AxyF, FxC, REPR::n, REPR::c, REPR::c, n_equations) +
                Multiply(AyyF, FyC, REPR::n, REPR::c, REPR::c, n_equations);

    return std::make_tuple(FxC, FyC, AFxC, AFyC);
}

/// @brief Backward Euler evolution equations in XY for fibers
template <typename VecT>
std::tuple<VecT, VecT> FiberEvolution(const VecT &AFxC, const VecT &AFyC, const FiberState<VecT> &Div,
                                      const FiberState<VecT> &oDiv, const VecT &UC, const VecT &VC, const double dt) {
    VecT eqXC = Div.XC_ - dt * AFxC - dt * UC - oDiv.XC_;
    VecT eqYC = Div.YC_ - dt * AFyC - dt * VC - oDiv.YC_;
    return std::make_tuple(eqXC, eqYC);
}

/// @brief Tension equation for penalty in our new system
template <typename VecT>
VecT FiberPenaltyTension(const FiberState<VecT> &Div, const FiberState<VecT> &oDiv, const VecT &UsC, const VecT &VsC,
                         const VecT &nUsC, const VecT &nVsC, const double dt, const unsigned int n_equations_tension) {
    // Eigen::IOFormat ColumnAsRowFmt(Eigen::StreamPrecision, 0, ",", ",", "", "", "[", "]");
    // Grab the proper namespaces for things like multiply, REPR
    using skelly_chebyshev::F2C;
    using skelly_chebyshev::Multiply;
    using skelly_chebyshev::REPR;

    // Alias our variables to make reading easier
    const VecT &XssssC = Div.XssssC_;
    const VecT &XsssC = Div.XsssC_;
    const VecT &XsC = Div.XsC_;
    const VecT &YssssC = Div.YssssC_;
    const VecT &YsssC = Div.YsssC_;
    const VecT &YsC = Div.YsC_;

    const VecT &TssC = Div.TssC_;
    const VecT &TC = Div.TC_;

    const VecT &oXsssC = oDiv.XsssC_;
    const VecT &oXssC = oDiv.XssC_;
    const VecT &oXsC = oDiv.XsC_;
    const VecT &oYsssC = oDiv.YsssC_;
    const VecT &oYssC = oDiv.YssC_;
    const VecT &oYsC = oDiv.YsC_;

    VecT WXC = 7.0 * Multiply(oXssC, XssssC, REPR::c, REPR::c, REPR::c, n_equations_tension) +
               6.0 * Multiply(oXsssC, XsssC, REPR::c, REPR::c, REPR::c, n_equations_tension);
    VecT WYC = 7.0 * Multiply(oYssC, YssssC, REPR::c, REPR::c, REPR::c, n_equations_tension) +
               6.0 * Multiply(oYsssC, YsssC, REPR::c, REPR::c, REPR::c, n_equations_tension);
    VecT W1C = Multiply(oXssC, oXssC, REPR::c, REPR::c, REPR::c, n_equations_tension) +
               Multiply(oYssC, oYssC, REPR::c, REPR::c, REPR::c, n_equations_tension);
    VecT WUC = Multiply(UsC, oXsC, REPR::c, REPR::c, REPR::c, n_equations_tension);
    VecT WVC = Multiply(VsC, oYsC, REPR::c, REPR::c, REPR::c, n_equations_tension);
    VecT W2C = WUC + WVC;
    VecT W3F = Multiply(oXsC, XsC, REPR::c, REPR::c, REPR::n, n_equations_tension) +
               Multiply(oYsC, YsC, REPR::c, REPR::c, REPR::n, n_equations_tension) - VecT::Ones(n_equations_tension);
    VecT W3C = F2C(W3F);

    // Put the overall factor of dt in the prefactor
    VecT WTC = Multiply(TC, W1C, REPR::c, REPR::c, REPR::c, n_equations_tension);
    VecT eqTC = 2.0 * TssC.array() - WTC.array() + WXC.array() + WYC.array() + W2C.array() + W3C.array() / dt;
    return eqTC;
}

/// @brief Clamped boundary condition
template <typename T, typename VecT>
FiberBoundaryCondition<T> ClampedBC(const FiberState<VecT> &Div, const FiberState<VecT> &oDiv, FSIDE side,
                                    const VecT &ClampPosition, const VecT &ClampDirector) {
    using skelly_chebyshev::LeftEvalPoly;
    using skelly_chebyshev::RightEvalPoly;
    // Evaluate at the proper place (left, right), use ternay operators to do this
    auto XsssC = (side == FSIDE::left) ? LeftEvalPoly<T, VecT>(Div.XsssC_) : RightEvalPoly<T, VecT>(Div.XsssC_);
    auto XsC = (side == FSIDE::left) ? LeftEvalPoly<T, VecT>(Div.XsC_) : RightEvalPoly<T, VecT>(Div.XsC_);
    auto XC = (side == FSIDE::left) ? LeftEvalPoly<T, VecT>(Div.XC_) : RightEvalPoly<T, VecT>(Div.XC_);
    auto YsssC = (side == FSIDE::left) ? LeftEvalPoly<T, VecT>(Div.YsssC_) : RightEvalPoly<T, VecT>(Div.YsssC_);
    auto YsC = (side == FSIDE::left) ? LeftEvalPoly<T, VecT>(Div.YsC_) : RightEvalPoly<T, VecT>(Div.YsC_);
    auto YC = (side == FSIDE::left) ? LeftEvalPoly<T, VecT>(Div.YC_) : RightEvalPoly<T, VecT>(Div.YC_);

    auto TsC = (side == FSIDE::left) ? LeftEvalPoly<T, VecT>(Div.TsC_) : RightEvalPoly<T, VecT>(Div.TsC_);

    auto oXssC = (side == FSIDE::left) ? LeftEvalPoly<T, VecT>(oDiv.XssC_) : RightEvalPoly<T, VecT>(oDiv.XssC_);
    auto oYssC = (side == FSIDE::left) ? LeftEvalPoly<T, VecT>(oDiv.YssC_) : RightEvalPoly<T, VecT>(oDiv.YssC_);

    auto W1 = XsssC * oXssC + YsssC * oYssC;

    return FiberBoundaryCondition<T>(XC - ClampPosition[0], XsC - ClampDirector[0], YC - ClampPosition[1],
                                     YsC - ClampDirector[1], TsC + 3 * W1);
}

/// @brief Free boundary condition
template <typename T, typename VecT>
FiberBoundaryCondition<T> FreeBC(const FiberState<VecT> &Div, FSIDE side) {
    using skelly_chebyshev::LeftEvalPoly;
    using skelly_chebyshev::RightEvalPoly;

    auto X1 = (side == FSIDE::left) ? LeftEvalPoly<T, VecT>(Div.XssC_) : RightEvalPoly<T, VecT>(Div.XssC_);
    auto X2 = (side == FSIDE::left) ? LeftEvalPoly<T, VecT>(Div.XsssC_) : RightEvalPoly<T, VecT>(Div.XsssC_);
    auto Y1 = (side == FSIDE::left) ? LeftEvalPoly<T, VecT>(Div.YssC_) : RightEvalPoly<T, VecT>(Div.YssC_);
    auto Y2 = (side == FSIDE::left) ? LeftEvalPoly<T, VecT>(Div.YsssC_) : RightEvalPoly<T, VecT>(Div.YsssC_);
    auto Tension = (side == FSIDE::left) ? LeftEvalPoly<T, VecT>(Div.TC_) : RightEvalPoly<T, VecT>(Div.TC_);

    return FiberBoundaryCondition<T>(X1, X2, Y1, Y2, Tension);
}

/// @brief Combine X equations with boundary conditions
template <typename T, typename VecT>
inline VecT CombineXWithBCs(const VecT &eq, const FiberBoundaryCondition<T> &BCL,
                            const FiberBoundaryCondition<T> &BCR) {
    VecT eqX(eq.size() + 4);
    eqX << eq, BCL.X1_, BCL.X2_, BCR.X1_, BCR.X2_;
    return eqX;
}

/// @brief Combine Y equations with boundary conditions
template <typename T, typename VecT>
inline VecT CombineYWithBCs(const VecT &eq, const FiberBoundaryCondition<T> &BCL,
                            const FiberBoundaryCondition<T> &BCR) {
    VecT eqY(eq.size() + 4);
    eqY << eq, BCL.Y1_, BCL.Y2_, BCR.Y1_, BCR.Y2_;
    return eqY;
}

/// @brief Combine T equations with boundary conditions for penalty type fibers
template <typename T, typename VecT>
inline VecT CombineTPenaltyWithBCs(const VecT &eq, const FiberBoundaryCondition<T> &BCL,
                                   const FiberBoundaryCondition<T> &BCR) {
    // Tension is slightly different than the X and Y components
    VecT eqT(eq.size() + 2);
    eqT << eq, BCL.T_, BCR.T_;
    return eqT;
}

} // namespace skelly_fiber

#endif // SKELLY_FIBER_HPP_
