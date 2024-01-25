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
    //
    // Include all of the derivatives, as we usually want to keep them around
    Eigen::VectorXd XX_;     ///< concatenated entire state of system
    Eigen::VectorXd XssssC_; ///< Fourth derivative of X components in coefficient space
    Eigen::VectorXd XsssC_;  ///< Third derivative of X components in coefficient space
    Eigen::VectorXd XssC_;   ///< Second derivative of X components in coefficient space
    Eigen::VectorXd XsC_;    ///< First derivative of X components in coefficient space
    Eigen::VectorXd XC_;     ///< Zeroth derivative of X components in coefficient space
    Eigen::VectorXd YssssC_; ///< Fourth derivative of Y components in coefficient space
    Eigen::VectorXd YsssC_;  ///< Third derivative of Y components in coefficient space
    Eigen::VectorXd YssC_;   ///< Second derivative of Y components in coefficient space
    Eigen::VectorXd YsC_;    ///< First derivative of Y components in coefficient space
    Eigen::VectorXd YC_;     ///< Zeroth derivative of Y components in coefficient space
    Eigen::VectorXd TssC_;   ///< Second derivative of T components in coefficient space
    Eigen::VectorXd TsC_;    ///< First derivative of T components in coefficient space
    Eigen::VectorXd TC_;     ///< Zeroth derivative of T components in coefficient space

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
    // instead, just have a function we always call to get the segment out of the main state vector. Icky. This
    // is done here via the 'Ref' constructor, but could also use 'VectorBlock', except later when composing
    // operations together.
    //
    // Get the different spatial and tension components of the state vector
    //
    // XXX could probably just be helper functions ane not in the class
    VectorRef XW() { return XX_.segment(0, n_nodes_); }
    VectorRef YW() { return XX_.segment(n_nodes_, n_nodes_); }
    VectorRef TW() { return XX_.segment(2 * n_nodes_, n_nodes_tension_); }

    // SplitXN an Eigen::Ref into different components
    // XXX Could probably be helper functions and not in the class
    std::tuple<VectorRef, VectorRef> SplitAt(VectorRef x, unsigned int n) {
        return std::make_tuple(x.segment(0, n), x.segment(n, x.size() - n));
    }

    // Divide and construct all derivatives of fiber state vector XX
    void DivideAndConstruct(double L) {
        // Divide up the main state array into its components
        Eigen::VectorXd Ax;
        Eigen::VectorXd Ay;
        Eigen::VectorXd At;
        std::tie(XssssC_, Ax) = SplitAt(XW(), n_equations_);
        std::tie(YssssC_, Ay) = SplitAt(YW(), n_equations_);
        std::tie(TsC_, At) = SplitAt(TW(), n_equations_tension_);

        // Now integrate up our solution
        // Get the integration matrices (in order) as we might have different ones later
        //
        // XXX should probably do this as a variadic template so that we can integrate an arbitrary number of this based
        // on what we provide.
        double rat = L / 2.0;
        IntegrateUp(XsssC_, XssC_, XsC_, XC_, XssssC_, rat, Ax[0], Ax[1], Ax[2], Ax[3]);
        IntegrateUp(YsssC_, YssC_, YsC_, YC_, YssssC_, rat, Ay[0], Ay[1], Ay[2], Ay[3]);
        TensionIntegrateUpConstraint(TC_, TsC_, IMT_, rat, At[0]);
    }

    // Do all 4 integrals for fibers
    //
    // Takes in references to the actual vector locations in order to fill them with data.
    //
    // XXX Could probably be helper functions and not in the class.
    //
    // 4th -> 3rd derivative
    void IntegrateUp(Eigen::VectorXd &XsssC, Eigen::VectorXd &XssC, Eigen::VectorXd &XsC, Eigen::VectorXd &XC,
                     CVectorRef &XssssC, const double rat, const double Ax, const double Bx, const double Cx,
                     const double Dx) {
        XsssC = (IM_ * XssssC) * rat;
        XsssC[0] += 6.0 * Dx;
        IntegrateUp(XssC, XsC, XC, XsssC, rat, Ax, Bx, Cx);
    }
    // 3rd -> 2nd derivative
    void IntegrateUp(Eigen::VectorXd &XssC, Eigen::VectorXd &XsC, Eigen::VectorXd &XC, CVectorRef &XsssC,
                     const double rat, const double Ax, const double Bx, const double Cx) {
        XssC = (IM_ * XsssC) * rat;
        XssC[0] += 2.0 * Cx;
        IntegrateUp(XsC, XC, XssC, rat, Ax, Bx);
    }
    // 2nd -> 1st derivative
    void IntegrateUp(Eigen::VectorXd &XsC, Eigen::VectorXd &XC, CVectorRef &XssC, const double rat, const double Ax,
                     const double Bx) {
        XsC = (IM_ * XssC) * rat;
        XsC[0] += Bx;
        IntegrateUp(XC, XsC, rat, Ax);
    }
    // 1st -> 0th derivative
    void IntegrateUp(Eigen::VectorXd &XC, CVectorRef &XsC, const double rat, const double Ax) {
        XC = (IM_ * XsC) * rat;
        XC[0] += Ax;
    }

    // Tension integrals
    //
    // XXX This should integrate up our tension conditions, but only for the CONSTRAINT version right now. Should make
    // this more general as well
    void TensionIntegrateUpConstraint(Eigen::VectorXd &TC, CVectorRef &TsC, CMatrixRef &IMT, const double rat,
                                      const double At) {
        TC = (IMT * TsC) * rat;
        TC[0] += At;
    }
};

#endif // FIBER_BASE_HPP_
