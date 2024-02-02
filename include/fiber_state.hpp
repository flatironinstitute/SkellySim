// @HEADER
// @HEADER

#ifndef FIBER_STATE_HPP_
#define FIBER_STATE_HPP_

/// \file fiber_state.hpp
/// \brief Container class for fiber state vector and associated derivatives/integrals
///
/// This is the state for a fiber, but is probably worth making general. Equivalent of 'Div' in David's code.

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

template <typename VecT>
class FiberState {
  public:
    // Input parameters
    unsigned int n_dim_;               ///< number of dimensions
    unsigned int n_nodes_;             ///< number of nodes representing XYZ
    unsigned int n_nodes_tension_;     ///< number of nodes representing tension
    unsigned int n_equations_;         ///< number of equations to use for XYZ
    unsigned int n_equations_tension_; ///< number of equations to use for tension

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

    /// @brief Construct a fiber of a given discretization
    FiberState(int n_dim, int n_nodes, int n_nodes_tension, int n_equations, int n_equations_tension)
        : n_dim_(n_dim), n_nodes_(n_nodes), n_nodes_tension_(n_nodes_tension), n_equations_(n_equations),
          n_equations_tension_(n_equations_tension) {
        // Set the main state vectors to their proper size and zero out. Have to create them explicitly, otherwise,
        // later VectorRef operations will cause segfaults
        XX_ = Eigen::VectorXd::Zero(n_dim_ * n_nodes_ + n_nodes_tension_);
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
        TssC_ = Eigen::VectorXd::Zero(n_equations_tension_);
        TsC_ = Eigen::VectorXd::Zero(n_equations_tension_);
        TC_ = Eigen::VectorXd::Zero(n_equations_tension_);
    }

    // Write to console
    friend auto operator<<(std::ostream &os, const FiberState<VecT> &m) -> std::ostream & {
        Eigen::IOFormat ColumnAsRowFmt(Eigen::StreamPrecision, 0, ",", ",", "", "", "[", "]");
        os << "XssssC:  " << m.XssssC_.format(ColumnAsRowFmt) << std::endl;
        os << "XsssC:   " << m.XsssC_.format(ColumnAsRowFmt) << std::endl;
        os << "XssC:    " << m.XssC_.format(ColumnAsRowFmt) << std::endl;
        os << "XsC:     " << m.XsC_.format(ColumnAsRowFmt) << std::endl;
        os << "XC:      " << m.XC_.format(ColumnAsRowFmt) << std::endl;

        os << "YssssC:  " << m.YssssC_.format(ColumnAsRowFmt) << std::endl;
        os << "YsssC:   " << m.YsssC_.format(ColumnAsRowFmt) << std::endl;
        os << "YssC:    " << m.YssC_.format(ColumnAsRowFmt) << std::endl;
        os << "YsC:     " << m.YsC_.format(ColumnAsRowFmt) << std::endl;
        os << "YC:      " << m.YC_.format(ColumnAsRowFmt) << std::endl;

        os << "TssC:    " << m.TssC_.format(ColumnAsRowFmt) << std::endl;
        os << "TsC:     " << m.TsC_.format(ColumnAsRowFmt) << std::endl;
        os << "TC:      " << m.TC_.format(ColumnAsRowFmt);

        return os;
    }
};

#endif // FIBER_STATE_HPP_