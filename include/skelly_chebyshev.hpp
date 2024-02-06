// @HEADER
// @HEADER

#ifndef SKELLY_CHEBYSHEV_HPP_
#define SKELLY_CHEBYSHEV_HPP_

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
#include <optional>
#include <unordered_map>

// SkellySim libs
#include <skelly_sim.hpp>

namespace skelly_chebyshev {

// Overall
// Is it worth making all of the things CONST that should be?
// Is it worth trying to get by reference?

/// @brief Enum for the operation type (coefficient c or node n) space
///
/// Used for converting between the different coefficient and node representations
enum REPR { c, n };

/// @brief Chebyshev ratio function
/// @tparam T Type of bounds
/// @param lb Lower bound
/// @param ub Upper bound
/// @return  T Chebyshev ratio
template <typename T>
inline T chebyshev_ratio(T lb, T ub) {
    return (ub - lb) / 2.0;
}

/// @brief Chebyshev inverse ratio function
/// @tparam T Type of bounds
/// @param lb Lower bound
/// @param ub Upper bound
/// @return  T Chebyshev inverse ratio
template <typename T>
inline T inverse_chebyshev_ratio(T lb, T ub) {
    return 2.0 / (ub - lb);
}

// inline Eigen::VectorXd to_scaled_space(double lb, double ub, Eigen::VectorXd x) {
inline Eigen::VectorXd to_scaled_space(double lb, double ub, CVectorRef &x) {
    Eigen::ArrayXd scaled_ret = (x.array() + 1.0) * chebyshev_ratio(lb, ub) + lb;
    return scaled_ret;
}

/// @brief Chebyshev quadratures points, but reversed from traditional chebyshev order
/// @param order Order of Chebyshev quadrature points
/// @return Eigen::VectorXd Vector of Chebyshev points
Eigen::VectorXd ChebyshevTPoints(const unsigned int order) {
    // Construct the vector in Eigen
    Eigen::ArrayXd thetas = M_PI / 2.0 * (2.0 * Eigen::ArrayXd::LinSpaced(order, order, 1.0) - 1.0) / order;
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
/// @param CVectorRef& x Const vector reference of Chebyshev zeros
/// @param const unsigned int n Order of chebyshev to use
/// @return Eigen::MatrixXd Vandermonde matrix for associated Chebyshev points at given order
Eigen::MatrixXd vander_julia_chebyshev(CVectorRef &x, const unsigned int n) {
    Eigen::MatrixXd A(x.size(), n + 1);
    // Set the first column to ones
    A.col(0).setOnes();
    if (n > 0) {
        A.col(1) = x;
        for (int i = 2; i < n + 1; i++) {
            A.col(i) = A.col(i - 1).array() * 2.0 * x.array() - A.col(i - 2).array();
        }
    }

    return A;
}

/// @brief VandermondeMatrix construction for ChebyshevTPoints at specified order
/// @param const unsigned int order Order of Vandermonde matrix to create
/// @return Eigen::MatrixXd Vandermonde matrix of order order
Eigen::MatrixXd VandermondeMatrix(const unsigned int order) {
    return vander_julia_chebyshev(ChebyshevTPoints(order), order - 1);
}

/// @brief VandermondeMatrix construction for a chebyshev of vector x
/// @param const unsigned int order Order of Vandermonde matrix
/// @param CVectorRef& x Points to create matrix for
/// @return Eigen::MatrixXd Vandermonde matrix of order order for points x
Eigen::MatrixXd VandermondeMatrix(const unsigned int order, CVectorRef &x) {
    return vander_julia_chebyshev(x, order - 1);
}

/// @brief Inverse VandermondeMatrix construction for ChebyshevTPoints at specified order
/// @param const unsigned int order Order of inverse Vandermonde matrix
/// @param CVectorRef& x Points to create matrix for
/// @return Eigen::MatrixXd Inverse Vandermonde matrix of order order for points x
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
Eigen::MatrixXd ToggleRepresentation(CMatrixRef &op, REPR op_in, REPR op_out, REPR req_in, REPR req_out) {
    // Explicitly use MatrixXd, otherwise there are reports of it doing a shallow copy
    Eigen::MatrixXd nop = op;

    // Check the input types
    if (op_in == REPR::c && req_in == REPR::n) {
        nop = nop * InverseVandermondeMatrix(op.cols());
    } else if (op_in == REPR::n && req_in == REPR::c) {
        nop = nop * VandermondeMatrix(op.cols());
    }

    // Check the output types
    if (op_out == REPR::c && req_out == REPR::n) {
        nop = VandermondeMatrix(op.rows()) * nop;
    } else if (op_out == REPR::n && req_out == REPR::c) {
        nop = InverseVandermondeMatrix(op.rows()) * nop;
    }

    return nop;
}

/// @brief derivative from the julia library
/// @param CVectorRef& p Vector of coefficients to construct derivative from
/// @return Eigen::VectorXd Derivative of input vector in coefficient space
///
/// Implemented based on julia's polynomials. See julia source code for more details
Eigen::VectorXd derivative_julia_chebyshev(CVectorRef &p) {
    // Copy the input vector to be able to modify on the fly
    Eigen::VectorXd q = p.segment(1, p.size() - 1);
    // Get the size
    auto n = q.size();
    // Create the derivative/return
    Eigen::VectorXd der = Eigen::VectorXd::Zero(n);

    // This section is hard as there are indexing problems for the values to be taken care of for
    // base 0 versus base 1 indexing where we also use the index to compute things.
    // Julia for j in n:-1:3
    // j is set to be base-1 index, so when using as an index, shift back
    for (int j = n; j > 2; j--) {
        der[j - 1] = 2.0 * j * q[j - 1];
        q[j - 1 - 2] += j * q[j - 1] / (j - 2);
    }
    // These don't need the shifts
    if (n > 1) {
        der[1] = 4.0 * q[1];
    }
    der[0] = q[0];

    return der;
}

/// @brief Construct first Chebyshev derivative of Tn
/// @param const unsigned int n Order of polynomial
/// @return Eigen::VectorXd First Chebyshev derivative for polynomial of order n
///
/// NOTE: Will use later for higher order derivatives!
Eigen::VectorXd FirstDerivativeOfChebyshevTn(const unsigned int n) {
    Eigen::VectorXd q = Eigen::VectorXd::Zero(n + 1);
    q[q.size() - 1] = 1.0;
    return derivative_julia_chebyshev(q);
}

/// @brief Construct the Nth Chebyshev derivative of Tn
/// @param const unsigned int n Order of polynomial
/// @param const unsigned int D Number of derivatives
/// @return Eigen::VectorXd Nth Chebyshev derivative for polynomial of order n
///
/// This may be slightly inefficient since we are replacing the previous vector to keep a running
/// multiplication.
Eigen::VectorXd NthDerivativeOfChebyshevTn(const unsigned int n, const unsigned int D) {
    // Loop over the derivatives
    Eigen::VectorXd n_derivative = FirstDerivativeOfChebyshevTn(n);
    for (int i = 2; i < D + 1; i++) {
        // Eigen seems smart enough to prevent aliasing
        n_derivative = derivative_julia_chebyshev(n_derivative);
    }

    return n_derivative;
}

/// @brief Spectral derivative matrix
Eigen::MatrixXd DerivativeMatrix(const unsigned int n, const unsigned int D, REPR in_type = REPR::c,
                                 REPR out_type = REPR::c, const double scale_factor = 1.0) {
    Eigen::MatrixXd DM = Eigen::MatrixXd::Zero(n - D, n);

    // Start at column D
    for (int i = D; i < n; i++) {
        DM.block(0, i, i - (D - 1), 1) = NthDerivativeOfChebyshevTn(i, D);
    }

    // Scale by scale_factor to the power of the number of derivatives
    DM = DM * pow(scale_factor, D);
    return ToggleRepresentation(DM, REPR::c, REPR::c, in_type, out_type);
}

/// @brief Spectral integration matrix
Eigen::MatrixXd IntegrationMatrix(const unsigned int order, REPR in_type = REPR::c, REPR out_type = REPR::c,
                                  const double scale_factor = 1.0) {
    Eigen::MatrixXd DMat = DerivativeMatrix(order, 1, REPR::c, REPR::c, scale_factor);
    Eigen::MatrixXd VM = VandermondeMatrix(order, Eigen::VectorXd{{-1.0}});
    Eigen::MatrixXd A(DMat.rows() + VM.rows(), DMat.cols());
    // vcat operation
    A << DMat, VM;

    return ToggleRepresentation(A.inverse(), REPR::c, REPR::c, in_type, out_type);
}

// ****************************************************************************************************************
// New hotness with templating for different functions
// ****************************************************************************************************************

// Need a cache for the vandermonde (and inverse vandermonde) matrices
std::unordered_map<unsigned int, Eigen::MatrixXd> vandermonde_cache;
std::unordered_map<unsigned int, Eigen::MatrixXd> inverse_vandermonde_cache;
// Have a function that looks if the key exists and then return it if it does, construct if it doesn't
// XXX Could be replaced with 'contains' if we move to c++20
inline Eigen::MatrixXd getVM(unsigned int order) {
    auto vit = vandermonde_cache.find(order);
    if (vit != vandermonde_cache.end()) {
        // Found the key, return it
        return vit->second;
    }
    vandermonde_cache[order] = VandermondeMatrix(order);
    return vandermonde_cache[order];
}
// Get the inverse vandermonde matrix
inline Eigen::MatrixXd getIVM(unsigned int order) {
    auto ivit = inverse_vandermonde_cache.find(order);
    if (ivit != inverse_vandermonde_cache.end()) {
        // Found the key, return it
        return ivit->second;
    }
    inverse_vandermonde_cache[order] = InverseVandermondeMatrix(order);
    return inverse_vandermonde_cache[order];
}

/// @brief Convert from coefficient to function space specifying vandermonde matrix returning vector
template <typename VecT>
inline VecT C2F(const VecT &XC, CMatrixRef &VM) {
    return VM * XC;
}

/// @brief Convert from coefficient to function space returning vector
template <typename VecT>
inline VecT C2F(const VecT &XC) {
    return getVM(XC.size()) * XC;
}

/// @brief Convert from function to coefficient space specifying vandermonde matrix returning vector
template <typename VecT>
inline VecT F2C(const VecT &XF, CMatrixRef &IVM) {
    return IVM * XF;
}

/// @brief Convert from function to coefficient space returning vector
template <typename VecT>
inline VecT F2C(const VecT &XF) {
    return getIVM(XF.size()) * XF;
}

/// @brief Toggle the representation of a vector quantity
template <typename VecT>
inline VecT ToggleRepresentation(const VecT &X, REPR in_type, REPR out_type) {
    if (in_type == out_type) {
        return X;
    } else if (in_type == REPR::n) {
        return F2C(X);
    } else {
        return C2F(X);
    }
}

/// @brief Resize a vector for multiplication based on its representation
template <typename VecT>
inline VecT Resize(const VecT &X, const unsigned int n, REPR in_type, REPR out_type) {
    VecT WXO = Eigen::VectorXd::Zero(n);

    // Toggle representation to coefficient type
    VecT WX = ToggleRepresentation(X, in_type, REPR::c);
    // Do the resize
    if (n > X.size()) {
        WXO.segment(0, WX.size()) = WX;
    } else {
        WXO = WX.segment(0, WXO.size());
    }
    // Toggle representation
    VecT XO = ToggleRepresentation(WXO, REPR::c, out_type);

    return XO;
}

/// @brief Multiply two vectors together
template <typename VecT>
inline VecT Multiply(const VecT &X, const VecT &Y, REPR XT, REPR YT, REPR XYT,
                     const std::optional<unsigned int> &Nout = std::nullopt,
                     const std::optional<unsigned int> &NM = std::nullopt) {
    // Set defaults if we aren't passed them in
    unsigned int nin = std::max(X.size(), Y.size());
    unsigned int nout = Nout ? Nout.value() : nin;
    unsigned int nm = NM ? NM.value() : 2 * nin;

    VecT XR = Resize(X, nm, XT, REPR::n);
    VecT YR = Resize(Y, nm, YT, REPR::n);
    VecT XYR = XR.array() * YR.array();

    return Resize(XYR, nout, REPR::n, XYT);
}

/// @brief Evaluate function for Chebyshev series, assuming that the order is >= 2
template <typename T, typename VecT>
inline T evalpoly(const T &x, const VecT &ch) {
    T c0 = ch[ch.size() - 2];
    T c1 = ch[ch.size() - 1];

    for (int i = ch.size() - 3; i >= 0; i--) {
        T oldc0 = c0;
        c0 = ch[i] - c1;
        c1 = oldc0 + c1 * 2.0 * x;
    }

    return c0 + c1 * x;
}

/// @brief Evalulate Chebyshev Polynomial
///
/// Note, this is just reversed from evalpoly in terms of arguments
template <typename T, typename VecT>
inline T EvalPoly(const VecT &XC, const T &x) {
    return evalpoly(x, XC);
}

/// @brief Evaluate function for Chebyshev series at Left boundary (-1)
template <typename T, typename VecT>
inline T LeftEvalPoly(const VecT &X) {
    // Set this to the correct type
    T mone{-1.0};
    return EvalPoly(X, mone);
}

/// @brief Evaluate function for Chebyshev series at Right boundary (+1)
template <typename T, typename VecT>
inline T RightEvalPoly(const VecT &X) {
    // Set this to the correct type
    T one{1.0};
    return EvalPoly(X, one);
}

} // namespace skelly_chebyshev

#endif // SKELLY_CHEBYSHEV_HPP_
