// @HEADER
// @HEADER

#ifndef FIBER_BASE_HPP_
#define FIBER_BASE_HPP_

/// \file fiber_chebyshev_constraint.hpp
/// \brief Fibers via chebyshev constraint implementation (with integration)
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

class FiberChebyshevConstraint {
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
    // Similar to the state vectors, for the matrix form of the problem, we want to keep the matrices that are the
    // equivalent of performing the integration, but not actually apply them. However, for each fiber, these only need
    // to be constructed once for a given node and equation number.
    //
    // Could probably move into an static call so they are preserved across all the different instantiations of the
    // fiber and then can be used repeatedly.
    Eigen::MatrixXd XssssM_;
    Eigen::MatrixXd XsssM_;
    Eigen::MatrixXd XssM_;
    Eigen::MatrixXd XsM_;
    Eigen::MatrixXd XM_;
    Eigen::MatrixXd TssM_;
    Eigen::MatrixXd TsM_;
    Eigen::MatrixXd TM_;

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
    FiberChebyshevConstraint(int n_nodes, int n_nodes_tension, int n_equations, int n_equations_tension)
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

        XssssM_ = Eigen::MatrixXd::Zero(n_equations_, n_equations_ + 4);
        XsssM_ = Eigen::MatrixXd::Zero(n_equations_, n_equations_ + 4);
        XssM_ = Eigen::MatrixXd::Zero(n_equations_, n_equations_ + 4);
        XsM_ = Eigen::MatrixXd::Zero(n_equations_, n_equations_ + 4);
        XM_ = Eigen::MatrixXd::Zero(n_equations_, n_equations_ + 4);
        TssM_ = Eigen::MatrixXd::Zero(n_equations_tension_, n_equations_tension_ + 1);
        TsM_ = Eigen::MatrixXd::Zero(n_equations_tension_, n_equations_tension_ + 1);
        TM_ = Eigen::MatrixXd::Zero(n_equations_tension_, n_equations_tension_ + 1);
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
    void IntegrateUp(VectorRef XsssC, VectorRef XssC, VectorRef XsC, VectorRef XC, CVectorRef &XssssC, const double rat,
                     const double Ax, const double Bx, const double Cx, const double Dx) {
        XsssC = (IM_ * XssssC) * rat;
        XsssC[0] += 6.0 * Dx;
        IntegrateUp(XssC, XsC, XC, XsssC, rat, Ax, Bx, Cx);
    }
    // 3rd -> 2nd derivative
    void IntegrateUp(VectorRef XssC, VectorRef XsC, VectorRef &XC, CVectorRef &XsssC, const double rat, const double Ax,
                     const double Bx, const double Cx) {
        XssC = (IM_ * XsssC) * rat;
        XssC[0] += 2.0 * Cx;
        IntegrateUp(XsC, XC, XssC, rat, Ax, Bx);
    }
    // 2nd -> 1st derivative
    void IntegrateUp(VectorRef XsC, VectorRef XC, CVectorRef &XssC, const double rat, const double Ax,
                     const double Bx) {
        XsC = (IM_ * XssC) * rat;
        XsC[0] += Bx;
        IntegrateUp(XC, XsC, rat, Ax);
    }
    // 1st -> 0th derivative
    void IntegrateUp(VectorRef XC, CVectorRef &XsC, const double rat, const double Ax) {
        XC = (IM_ * XsC) * rat;
        XC[0] += Ax;
    }

    // Create integration matrices for going from the XssssC derivative all the way up to Xc, but don't
    // apply them, as we will need them to solve later. Slightly different syntax as the IntegrateUp, as we are
    // going to use these operations to also extract information later. Construct all 5 of the matrices at the same
    // time.
    //
    // Note that this is for the Chebyshev fibers with constraint with a state vector that contains the fourth
    // derivatives.
    void ConstructIntegrationMatricesX(MatrixRef XssssM, MatrixRef XsssM, MatrixRef XssM, MatrixRef XsM, MatrixRef XM,
                                       CMatrixRef IM, const double rat) {
        // Need to construct from XssssM --> XM, going through the steps between
        // XssssM
        XssssM << Eigen::MatrixXd::Identity(n_equations_, n_equations_), Eigen::MatrixXd::Zero(n_equations_, 4);
        // XsssM (with boundary term now)
        Eigen::MatrixXd BC = Eigen::MatrixXd::Zero(n_equations_, 4);
        BC(0, 3) = 6.0;
        XsssM << (rat * IM), BC;
        // XssM
        XssM = (rat * IM) * XsssM;
        XssM(0, XssM.cols() - 1 - 1) = 2.0;
        // XsM
        XsM = (rat * IM) * XssM;
        XsM(0, XsM.cols() - 1 - 2) = 1.0;
        // XM
        XM = (rat * IM) * XsM;
        XM(0, XM.cols() - 1 - 3) = 1.0;
    }

    // Now do the tension integration matrices
    void ConstructIntegrationMatricesT(MatrixRef TsM, MatrixRef TM, CMatrixRef &IMT, const double rat) {
        // Need to get just the tension components
        TsM << Eigen::MatrixXd::Identity(n_equations_tension_, n_equations_tension_),
            Eigen::MatrixXd::Zero(n_equations_tension_, 1);
        Eigen::VectorXd BC = Eigen::VectorXd::Zero(n_equations_tension_ + 1);
        BC[0] = 1.0;
        TM << (rat * IMT), BC;
    }

    // Tension integrals
    //
    // XXX This should integrate up our tension conditions, but only for the CONSTRAINT version right now. Should make
    // this more general as well
    void TensionIntegrateUpConstraint(VectorRef TC, CVectorRef &TsC, CMatrixRef &IMT, const double rat,
                                      const double At) {
        TC = (IMT * TsC) * rat;
        TC[0] += At;
    }
};

#endif // FIBER_BASE_HPP_
