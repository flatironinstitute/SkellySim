#ifndef FIBER_CONTAINER_HPP
#define FIBER_CONTAINER_HPP

#include <skelly_sim.hpp>

#include <Eigen/LU>
#include <list>
#include <unordered_map>

#include <kernels.hpp>
#include <params.hpp>

class Periphery;
class BodyContainer;

/// Class to hold the fiber objects.
///
/// The container object is designed to work on fibers local to that MPI rank. Each MPI rank
/// should have its own container, with its own unique fibers. The container object does not
/// have any knowledge of the MPI world state, which, for example, is passed in externally to
/// the FiberContainer::flow method and potentially others.
///
/// Developer note: ideally all interactions with the fiber objects should be through this
/// container, except for testing purposes. Operating on fibers outside of the container class
/// is ill-advised.
class FiberContainer {
  public:
    std::vector<Fiber> fibers; ///< Array of fibers local to this MPI rank
    /// pointer to FMM object (pointer to avoid constructing stokeslet_kernel_ with default FiberContainer)
    kernels::Evaluator stokeslet_kernel_;
    // std::shared_ptr<kernels::FMM<stkfmm::Stk3DFMM>> stokeslet_kernel_;

    /// Empty container constructor to avoid initialization list complications. No way to
    /// initialize after using this constructor, so overwrite objects with full constructor.
    FiberContainer() = default;
    FiberContainer(Params &params);
    FiberContainer(toml::array &fiber_tables, Params &params);

    void update_derivatives();
    void update_stokeslets(double eta);
    void update_linear_operators(double dt, double eta);
    void update_cache_variables(double dt, double eta);
    void update_local_node_positions();
    void update_RHS(double dt, MatrixRef &v_on_fibers, MatrixRef &f_on_fibers);
    void apply_bc_rectangular(double dt, MatrixRef &v_on_fibers, MatrixRef &f_on_fibers);
    void step(VectorRef &fiber_sol);
    void repin_to_bodies(BodyContainer &bodies);
    void set_evaluator(const std::string &evaluator);

    /// @brief get total number of nodes across fibers in the container
    /// Usually you need this to form arrays used as input later
    /// @returns total number of nodes across fibers in the container :)
    int get_local_node_count() const {
        // FIXME: This could certainly be cached
        int tot = 0;
        for (auto &fib : fibers)
            tot += fib.n_nodes_;
        return tot;
    };

    int get_global_total_fib_nodes() const;

    /// @brief Get the size of all local fibers contribution to the matrix problem solution
    int get_local_solution_size() const { return get_local_node_count() * 4; }

    /// @brief Get number of local fibers
    int get_local_count() const { return fibers.size(); };

    int get_global_count() const;

    Eigen::MatrixXd generate_constant_force() const;
    const Eigen::MatrixXd &get_local_node_positions() const { return r_fib_local_; };
    Eigen::VectorXd get_RHS() const;
    Eigen::MatrixXd flow(const MatrixRef &r_trg, const MatrixRef &forces, double eta, bool subtract_self = true) const;
    Eigen::VectorXd matvec(VectorRef &x_all, MatrixRef &v_fib, MatrixRef &v_fib_boundary) const;
    Eigen::MatrixXd apply_fiber_force(VectorRef &x_all) const;
    Eigen::VectorXd apply_preconditioner(VectorRef &x_all) const;

    void update_boundary_conditions(Periphery &shell, const periphery_binding_t &periphery_binding);

    ActiveIterator<Fiber> begin() { return ActiveIterator<Fiber>(0, fibers); }
    ActiveIterator<Fiber> end() { return ActiveIterator<Fiber>(fibers.size(), fibers); }

    ActiveIterator<Fiber> begin() const { return ActiveIterator<Fiber>(0, const_cast<decltype(fibers) &>(fibers)); }
    ActiveIterator<Fiber> end() const {
        return ActiveIterator<Fiber>(fibers.size(), const_cast<decltype(fibers) &>(fibers));
    }

  private:
    int world_size_ = -1;
    int world_rank_;
    Eigen::MatrixXd r_fib_local_;

  public:
    MSGPACK_DEFINE(fibers);
};

#endif
