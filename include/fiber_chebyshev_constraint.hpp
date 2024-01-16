// @HEADER
// @HEADER

#ifndef FIBER_CHEBYSHEV_CONSTRAINT
#define FIBER_CHEBYSHEV_CONSTRAINT

/// \file fiber_chebyshev_constraint.hpp
/// \brief Class to represent a single flexible filament with Chebyshev spectral methods and tension constraints
///
/// Actions on the fiber class are typically handled via the container object, which will
/// distribute calls appropriately across all fibers in the container.

// External libs
#include <Eigen/LU>

// C++ core libs
#include <list>
#include <unordered_map>

// SkellySim libs
#include <kernels.hpp>
#include <params.hpp>
#include <skelly_sim.hpp>

class Periphery;
class BodyContainer;

class FiberChebyshevConstraint {
  public:
    enum BC { Force, Torque, Velocity, AngularVelocity, Position, Angle };
    static const std::string BC_name[];

    // Input parameters
    int n_nodes_;             ///< number of nodes representing the fiber for XYZ
    int n_nodes_tension_;     ///< number of nodes representing fiber fo tension
    int n_equations_;         ///< number of equations to use for XYZ
    int n_equations_tension_; ///< number of equations to use for tension
    double radius_;      ///< radius of the fiber (for slender-body-theory, though possibly for collisions eventually)
    double length_;      ///< Desired 'constraint' length of fiber
    double length_prev_; ///< Last accepted length_
    double bending_rigidity_; ///< bending rigidity 'E' of fiber

    /// @brief Initialize an empty fiber
    /// @param[in] n_nodes fiber 'resolution'
    /// @param[in] radius fiber radius
    /// @param[in] length fiber length
    /// @param[in] bending_rigidity bending rigidity of fiber
    /// @param[in] eta fluid viscosity
    ///
    /// @deprecated Initializing with a toml::table structure is the preferred initialization. This is only around for
    /// testing.
    FiberChebyshevConstraint(int n_nodes, int n_nodes_tension, int n_equations, int n_equations_tension, double radius,
                             double length, double bending_rigidity, double eta)
        : n_nodes_(n_nodes), n_nodes_tension_(n_nodes_tension), n_equations_(n_equations),
          n_equations_tension_(n_equations_tension), radius_(radius), length_(length),
          bending_rigidity_(bending_rigidity) {}

    /// @brief Initialize values and resize arrays
    ///
    /// _MUST_ be called from constructors.
    ///
    /// Initializes: FiberChebyshevConstraint::x_, FiberChebyshevConstraint::xs_,
};

#endif // FIBER_CHEBYSHEV_CONSTRAINT_HPP_
