// @HEADER
// @HEADER

#ifndef FIBER_BASE_HPP_
#define FIBER_BASE_HPP_

/// \file fiber_base.hpp
/// \brief Base class for new fiber implementations
///
/// New Fiber base class to implement all of the different Chebyshev/spectral fibers

// External libs
#include <Eigen/LU>

// C libs
#include <math.h>

// C++ core libs
#include <iostream>

// SkellySim libs
#include <skelly_chebyshev.hpp>
#include <skelly_sim.hpp>

class FiberBase {
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
    Eigen::VectorXd XX_; ///< concatenated entire state of system

    /// @brief Construct a fiber of a given discretizaiton
    FiberBase(int n_nodes, int n_nodes_tension, int n_equations, int n_equations_tension)
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

        // Set the main state vector to the proper size and zero it out
        XX_ = Eigen::VectorXd(2 * n_nodes_ + n_nodes_tension_);
        XX_.setZero();
    }

    // Views
    //
    // Eigen is lame and we can't keep around a view as an object forever without knowing it's size, it seems. So,
    // instead, just have a function we always call to get the segment out of the main state vector. Icky.
    //
    // Get the different spatial and tension components of the state vector
    Eigen::VectorBlock<Eigen::VectorXd> XW() { return XX_.segment(0, n_nodes_); }
    Eigen::VectorBlock<Eigen::VectorXd> YW() { return XX_.segment(n_nodes_, n_nodes_); }
    Eigen::VectorBlock<Eigen::VectorXd> TW() { return XX_.segment(2 * n_nodes_, n_nodes_tension_); }
};

#endif // FIBER_BASE_HPP_
