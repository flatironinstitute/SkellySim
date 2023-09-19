#ifndef FIBER_CONTAINER_BASE_HPP
#define FIBER_CONTAINER_BASE_HPP

#include <kernels.hpp>
#include <params.hpp>
#include <skelly_sim.hpp>

class Periphery;
class BodyContainer;

/// Abstract base class to represent a container holding fiber objects
///
/// This is done so that there is a common interface for how to interact with fiber objects that
/// are local to each MPI rank.

class FiberContainerBase {
  public:
    /// Enum for the different fiber types we know about
    enum FIBERTYPE { None, FiniteDifference };

    //! \name Constructors and destructors
    //@{

    /// Empty container constructor to avoid initializing list complications. No way to initialize
    /// after using this constructor, so overwrite objects with full constructor.
    FiberContainerBase() = default;
    /// Constructor using TOML and params
    FiberContainerBase(toml::array &fiber_tables, Params &params);

    //@}

    //! \name Public member functions
    //@{

    /// @brief Set the pair interaction evaluator
    void set_evaluator(const std::string &evaluator);

    /// @brief Get the local number of fibers in the container
    int get_local_fiber_number() const { return n_local_fibers_; }

    /// @brief Get the local number of nodes across fibers in the container
    int get_local_node_count() const { return n_local_node_count_; }

    /// @brief Get the local solution size across fibers in this container
    int get_local_solution_size() const { return n_local_solution_size_; }

    /// @brief Get the global number of fibers in all MPI ranks
    int get_global_fiber_number() const;

    /// @brief Get the global number of nodes for fibers in all MPI ranks
    int get_global_node_count() const;

    /// @brief Get the global solution size across fibers in all MPI ranks
    int get_global_solution_size() const;

    /// @brief Set the local fiber number, node count, and solution size
    void set_local_fiber_numbers(int n_fibers, int n_local_nodes, int n_local_solution_size);

    /// @brief Get the local node positions in REAL SPACE
    const Eigen::MatrixXd &get_local_node_positions() const { return r_fib_local_; }

    //@}

    //! \name Public virtual functions
    //@{

    /// @brief initialize the fiber container, needs to be overridden by inherited classes
    virtual void init_fiber_container(toml::array &fiber_tables, Params &params) {
        throw std::runtime_error("init_fiber_container undefined on base FiberContainer class\n");
    }

    /// @brief Update the local node positions
    ///
    /// Most fibers have nodes, so this is a virtual method
    virtual void update_local_node_positions() {
        throw std::runtime_error("update_local_node_positions undefined on base FiberContainer class\n");
    }

    /// @brief Update any fiber cache variables
    virtual void update_cache_variables(double dt, double eta) {
        throw std::runtime_error("update_cache_variables undefined on base FiberContainer class\n");
    }

    /// @brief Generate a constant force
    virtual Eigen::MatrixXd generate_constant_force() const {
        throw std::runtime_error("generate_constant_force undefined on base FiberContainer class\n");
    }

    /// @brief Generate the flow
    virtual Eigen::MatrixXd flow(const MatrixRef &r_trg, const MatrixRef &forces, double eta,
                                 bool subtract_self = true) const {
        throw std::runtime_error("flow undefined on base FiberContainer class\n");
    }

    /// @brief Matvec operator
    virtual Eigen::VectorXd matvec(VectorRef &x_all, MatrixRef &v_fib, MatrixRef &v_fib_boundary) const {
        throw std::runtime_error("matvec undefined on base FiberContainer class\n");
    }

    /// @brief Update the RHS of the equation
    virtual void update_rhs(double dt, MatrixRef &v_on_fibers, MatrixRef &f_on_fibers) {
        throw std::runtime_error("update_rhs undefined on base FiberContainer class\n");
    }

    /// @brief Update the boundary conditions
    virtual void update_boundary_conditions(Periphery &shell, const periphery_binding_t &periphery_binding) {
        throw std::runtime_error("update_boundary_conditions undefined on base FiberContainer class\n");
    }

    /// @brief Apply the boundary conditions of the system
    virtual void apply_bcs(double dt, MatrixRef &v_on_fibers, MatrixRef &f_on_fibers) {
        throw std::runtime_error("apply_bcs undefined on base FiberContainer class\n");
    }

    /// @brief Apply preconditioner
    virtual Eigen::VectorXd apply_preconditioner(VectorRef &x_all) const {
        throw std::runtime_error("apply_preconditioner undefined on base FiberContainer class\n");
    }

    /// @brief Apply fiber force
    virtual Eigen::MatrixXd apply_fiber_force(VectorRef &x_all) const {
        throw std::runtime_error("apply_fiber_force undefined on base FiberContainer class\n");
    }

    /// @brief Perform a timestep
    virtual void step(VectorRef &fiber_sol) {
        throw std::runtime_error("step undefined on base FiberContainer class\n");
    }

    /// @brief Repin the fibers to bodies
    /// XXX FIXME This could be restated differently and abstracted more cleanly
    virtual void repin_to_bodies(BodyContainer &bodies) {
        throw std::runtime_error("repin_to_bodies undefined on base FiberContainer class\n");
    }

    //@}

    //! \name Public member variables
    //@{

    int n_local_fibers_ = -1;        ///< Cached number of local fibers in this container
    int n_local_node_count_ = -1;    ///< Cached number of local nodes for fibers in this container
    int n_local_solution_size_ = -1; ///< Cached local solution size for fibers in this container

    FIBERTYPE fiber_type_ = FIBERTYPE::None; ///< Fiber type (None to start with)

    /// Pointer to FMM object (pointer to avoid constructing stokeslet_kernel_ wtih default FiberContainer)
    kernels::Evaluator stokeslet_kernel_;

    //@}

  protected:
    int world_size_ = -1;
    int world_rank_;
    Eigen::MatrixXd r_fib_local_; ///< Local fiber node positions
};

#endif
