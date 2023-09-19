#ifndef FIBER_CONTAINER_FINITEDIFFERENCE_HPP
#define FIBER_CONTAINER_FINITEDIFFERENCE_HPP

#include <fiber_container_base.hpp>
#include <fiber_finitedifference.hpp>
#include <kernels.hpp>
#include <params.hpp>
#include <skelly_sim.hpp>

class Periphery;
class BodyContainer;

/// Finite difference fiber container class

class FiberContainerFinitedifference : public FiberContainerBase {
  public:
    //! \name Constructors and destructors
    //@{

    /// Constructor using TOML and params
    FiberContainerFinitedifference(toml::array &fiber_tables, Params &params);

    //@}

    //! \name Public member functions
    //@{

    /// Initialize the fiber container
    void init_fiber_container(toml::array &fiber_tables, Params &params) override;

    /// Update the local (MPI) node positions
    void update_local_node_positions() override;

    /// Update the cache variables
    void update_cache_variables(double dt, double eta) override;

    /// Generate a constant force
    Eigen::MatrixXd generate_constant_force() const override;

    /// Generate the flow
    Eigen::MatrixXd flow(const MatrixRef &r_trg, const MatrixRef &forces, double eta,
                         bool subtract_self = true) const override;

    /// Apply the matvec operator
    virtual Eigen::VectorXd matvec(VectorRef &x_all, MatrixRef &v_fib, MatrixRef &v_fib_boundary) const override;

    /// Update the RHS of the equation
    void update_rhs(double dt, MatrixRef &v_on_fibers, MatrixRef &f_on_fibers) override;

    /// Update the boundary conditions
    void update_boundary_conditions(Periphery &shell, const periphery_binding_t &periphery_binding) override;

    /// Apply boundary conditions
    void apply_bcs(double dt, MatrixRef &v_on_fibers, MatrixRef &f_on_fibers) override;

    /// Apply the preconditioner
    Eigen::VectorXd apply_preconditioner(VectorRef &x_all) const override;

    /// Apply fiber force
    virtual Eigen::MatrixXd apply_fiber_force(VectorRef &x_all) const override;

    /// Perform a timestep
    void step(VectorRef &fiber_sol) override;

    /// Repin the fibers to a body
    void repin_to_bodies(BodyContainer &bodies) override;

    /// Begin of an iterator to this object
    ActiveIterator<FiberFiniteDifference> begin() { return ActiveIterator<FiberFiniteDifference>(0, fibers_); }

    /// End of an iterator to this object
    ActiveIterator<FiberFiniteDifference> end() {
        return ActiveIterator<FiberFiniteDifference>(fibers_.size(), fibers_);
    }

    /// Begin of a const iterator to this object
    ActiveIterator<FiberFiniteDifference> begin() const {
        return ActiveIterator<FiberFiniteDifference>(0, const_cast<decltype(fibers_) &>(fibers_));
    }

    /// End of a const iterator to this object
    ActiveIterator<FiberFiniteDifference> end() const {
        return ActiveIterator<FiberFiniteDifference>(fibers_.size(), const_cast<decltype(fibers_) &>(fibers_));
    }

    //@}

    //! \name Public member variables
    //@{

    std::vector<FiberFiniteDifference> fibers_; ///< Array of fibers local to this MPI rank

    //@}

    //! \name Special external things
    MSGPACK_DEFINE(fibers_);

  private:
    //! \name Private member helper functions
    //@{

    /// Apply the rectangular BCs for a finite difference fiber
    void apply_bc_rectangular(double dt, MatrixRef &v_on_fibers, MatrixRef &f_on_fibers);

    //@}
};

#endif
