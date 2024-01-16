// @HEADER
// @HEADER

#ifndef SKELLY_CHEBYSSHEV
#define SKELLY_CHEBYSSHEV

/// \file skelly_chebyshev.hpp
/// \brief Functions for Chebyshev polynomials in SkellySim
///
/// Chebyshev polynomial functions derived from their equivalent form in Julia, to implement David
/// Stein's FiberTets.jl in c++

// External libs
#include <Eigen/LU>

// C libs
#include <math.h>

// C++ core libs
#include <iostream>

// SkellySim libs
#include <skelly_sim.hpp>

namespace skelly_chebyshev {

    // Overall
    // Is it worth making all of the things CONST that should be?
    // Is it worth trying to get by reference?

    /// @brief Enum for the operation type (coefficient c or node n) space
    ///
    /// Used for converting between the different coeffcient and node representations
    enum CTYPE { c, n };

    /// @brief Chebyshev ratio function
    /// @tparam T Type of bounds
    /// @param lb Lower bound
    /// @param ub Upper bound
    /// @return  T Chebyshev ratio
    template<typename T>
    inline T chebyshev_ratio(T lb, T ub) { return (ub-lb)/2.0; }

    /// @brief Chebyshev inverse ratio function
    /// @tparam T Type of bounds
    /// @param lb Lower bound
    /// @param ub Upper bound
    /// @return  T Chebyshev inverse ratio
    template<typename T>
    inline T inverse_chebyshev_ratio(T lb, T ub) { return 2.0/(ub-lb); }

    //inline Eigen::VectorXd to_scaled_space(double lb, double ub, Eigen::VectorXd x) {
    inline Eigen::VectorXd to_scaled_space(double lb, double ub, VectorRef& x) {
        Eigen::ArrayXd scaled_ret = (x.array()+1.0)*chebyshev_ratio(lb, ub) + lb;
        return scaled_ret;
    }

    /// @brief Chebyshev quadratures points, but reversed from traditional chebyshev order
    /// @param order Order of Chebyshev quadrature points
    /// @return Eigen::VectorXd Vector of Chebyshev points
    Eigen::VectorXd ChebyshevTPoints(const unsigned int order) {
        // Construct the vector in Eigen
        Eigen::ArrayXd thetas = M_PI/2.0*(2.0*Eigen::ArrayXd::LinSpaced(order, order, 1.0)-1.0) / order;
        return cos(thetas);
    }

    /// @brief Chebyshev quadratures points, but reversed from traditional chebyshev order
    /// @param double lb Lower bound
    /// @param double ub Upper bound
    /// @param order Order
    /// @return Eigen::VectorXd Vector of Chebyshev points
    Eigen::VectorXd ChebyshevTPoints(double lb, double ub, const unsigned int order) {
        // Create the Chebyshev points
        Eigen::VectorXd chebyshev_points = ChebyshevTPoints(order);
        return to_scaled_space(lb, ub, chebyshev_points);
    }

    /// @brief vander from the Julia Polynomials library specifically for chebyshev points
    Eigen::MatrixXd vander_julia_chebyshev(VectorRef& x, const unsigned int n) {
        Eigen::MatrixXd A(x.size(), n+1);
        // Set the first column to ones
        A.col(0).setOnes();
        if (n > 0) {
            A.col(1) = x;
            for (int i = 2; i < n+1; i++) {
                A.col(i) = A.col(i-1).array() * 2.0*x.array() - A.col(i-2).array();
            }
        }

        return A;
    }

    /// @brief VandermondeMatrix construction for ChebyshevTPoints at specified order
    Eigen::MatrixXd VandermondeMatrix(const unsigned int order) {
        return vander_julia_chebyshev(ChebyshevTPoints(order), order-1);
    }

    /// @brief VandermondeMatrix construction for a chebyshev of vector x
    Eigen::MatrixXd VandermondeMatrix(const unsigned int order, VectorRef& x) {
        return vander_julia_chebyshev(x, order-1);
    }

    /// @brief Inverse VandermondeMatrix construction for ChebyshevTPoints at specified order
    Eigen::MatrixXd InverseVandermondeMatrix(const unsigned int order) {
        Eigen::MatrixXd IVM = VandermondeMatrix(order);
        return IVM.inverse();
    }

    /// @brief Toggle the representation between coefficient and node space
    ///
    /// From David:
    /// Give a Chebyshev operator OP which acts from OPIN --> OPOUT,
    ///     create a new Chebyshev operator acting from REQIN --> REQOUT
    ///
    /// For example, if OP acts from nodes --> nodes, then:
    ///     NOP = ToggleRepresentation(OP, CTYPE::n, CTYPE::n, CTYPE::c, CTYPE::n)
    /// returns a new operator NOP that acts from coefficients --> nodees
    Eigen::MatrixXd ToggleRepresentation(MatrixRef& OP, CTYPE OPIN, CTYPE OPOUT, CTYPE REQUIN, CTYPE REQOUT) {
        // Explicity use MatrixXd, otherwise there are reports of it doing a shallow copy
        Eigen::MatrixXd NOP = OP;

        return NOP;
    }

    /// @brief Compute spectral derivative matrix
    Eigen::MatrixXd DerivativeMatrix(const unsigned int order,
                                     const unsigned int D,
                                     CTYPE in_type,
                                     CTYPE out_type,
                                     double scale_factor = 1.0) {
        Eigen::MatrixXd DM = Eigen::MatrixXd::Zero(order-D, order);

        return DM;
    }
}

#endif // SKELLY_CHEBYSSHEV
