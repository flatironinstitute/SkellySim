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

} // namespace skelly_fiber

#endif // SKELLY_FIBER_HPP_
