#ifndef FIBER_CONTAINER_FINITE_DIFFERENCE_HPP
#define FIBER_CONTAINER_FINITE_DIFFERENCE_HPP

#include <fiber_container_base.hpp>
#include <fiber_finite_difference.hpp>
#include <kernels.hpp>
#include <params.hpp>
#include <skelly_sim.hpp>

class Periphery;
class BodyContainer;

/// Finite difference fiber container class

class FiberContainerFiniteDifference : public FiberContainerBase {
  public:
    //! \name Constructors and destructors
    //@{

    /// Almost empty container for list initializations
    ///
    /// Need this for initializaitons of an empty container, but we need to have the FIBERTYPE set
    FiberContainerFiniteDifference();

    /// Constructor using TOML and params
    FiberContainerFiniteDifference(toml::array &fiber_tables, Params &params);

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

    /// Get the RHS of the solution for FiberFiniteDifference
    Eigen::VectorXd get_rhs() const override;

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

    int get_local_fiber_count() const override {
        // FIXME: DI: Doesn't account for inactive fibers.
        return fibers_.size();
    }
    int get_local_node_count() const override {
        // FIXME: Should cache this. Also won't work for DI (just change to fib: *this)
        int node_count = 0;
        for (auto &fib: *this)
            node_count += fib.n_nodes_;
        return node_count;
    }
    int get_local_solution_size() const override { return get_local_node_count() * 4; }

    double fiber_error_local() const override;

    bool check_collision(const Periphery &periphery, double threshold) const override;
    Eigen::MatrixXd periphery_force(const Periphery &, const fiber_periphery_interaction_t &) const override;

    //@}

    //! \name Public member variables
    //@{

    std::vector<FiberFiniteDifference> fibers_; ///< Array of fibers local to this MPI rank

    //@}

    //! \name Special external things
    MSGPACK_DEFINE(fiber_type_, fibers_);

  private:
    //! \name Private member helper functions
    //@{

    /// Apply the rectangular BCs for a finite difference fiber
    void apply_bc_rectangular(double dt, MatrixRef &v_on_fibers, MatrixRef &f_on_fibers);

    //@}
};

#endif
