#ifndef BODY_CONTAINER_HPP
#define BODY_CONTAINER_HPP

#include <skelly_sim.hpp>

#include <Eigen/LU>
#include <body_deformable.hpp>
#include <body_ellipsoidal.hpp>
#include <body_spherical.hpp>
#include <kernels.hpp>
#include <params.hpp>

class Periphery;
class Body;
class FiberContainerFiniteDifference;

/// @brief Container for multiple generic Body objects
class BodyContainer {
  private:
    int world_rank_ = 0; ///< MPI world rank
    int world_size_;     ///< MPI world size

  public:
    /// Vector of body pointers
    std::vector<std::shared_ptr<Body>> bodies;
    std::vector<std::shared_ptr<SphericalBody>> spherical_bodies;
    std::vector<std::shared_ptr<DeformableBody>> deformable_bodies;
    std::vector<std::shared_ptr<EllipsoidalBody>> ellipsoidal_bodies;
    std::unordered_map<std::shared_ptr<Body>, int> solution_offsets_;
    std::unordered_map<std::shared_ptr<Body>, int> node_offsets_;

    /// Pointer to FMM stresslet kernel (StokesDoubleLayer)
    kernels::Evaluator stresslet_kernel_;
    /// Pointer to FMM stokeslet kernel (Stokes)
    kernels::Evaluator stokeslet_kernel_;

    /// @brief Empty container constructor to avoid initialization list complications.
    BodyContainer() = default;
    BodyContainer(toml::array &body_tables, Params &params);

    // FIXME: remove redundant code in =/copy
    /// @brief Copy constructor...
    BodyContainer(const BodyContainer &orig);

    /// @brief Assignment operator...
    BodyContainer &operator=(const BodyContainer orig);

    void populate_sublists();

    template <typename T>
    Eigen::VectorXd get_local_solution(const T &body_vec, VectorRef &body_solutions) const;

    template <typename T>
    Eigen::MatrixXd get_local_node_positions(const T &body_vec) const;

    template <typename T>
    Eigen::MatrixXd get_local_node_normals(const T &body_vec) const;

    template <typename T>
    Eigen::MatrixXd get_local_center_positions(const T &body_vec) const;

    template <typename T>
    Eigen::MatrixXd get_global_center_positions(const T &body_vec) const;

    template <typename T>
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> get_global_forces_torques(const T &body_vec) const;

    Eigen::MatrixXd calculate_external_forces_torques(double time) const;

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
    calculate_link_conditions(VectorRef &fiber_sol, VectorRef &x_bodies,
                              const FiberContainerFiniteDifference &fc) const;

    /// @brief Get total number of nodes associated with body
    ///
    /// See BodyContainer::get_local_solution_size for notes about how this is distributed
    int get_local_node_count() const {
        if (world_rank_ != 0)
            return 0;

        int tot = 0;
        for (const auto &body : bodies)
            tot += body->n_nodes_;
        return tot;
    }

    /// @brief Get the size of all bodies contribution to the matrix problem solution, regardless of rank
    int get_global_solution_size() const {
        int sol_size = 0;
        for (const auto &body : bodies)
            sol_size += body->get_solution_size();
        return sol_size;
    }

    /// @brief Get the size of all local bodies contribution to the matrix problem solution
    ///
    /// Since there aren't many bodies, and there is no easy way to synchronize them across processes, the rank 0
    /// process handles all components of the solution.
    int get_local_solution_size() const { return world_rank_ == 0 ? get_global_solution_size() : 0; }

    void update_RHS(MatrixRef &v_on_body);
    Eigen::VectorXd get_RHS() const;

    Eigen::VectorXd matvec(MatrixRef &v_bodies, VectorRef &x_bodies) const;
    Eigen::VectorXd apply_preconditioner(VectorRef &X) const;
    Eigen::MatrixXd flow(MatrixRef &r_trg, VectorRef &body_solutions, MatrixRef &link_conditions, double eta) const;
    Eigen::MatrixXd flow_spherical(MatrixRef &r_trg, VectorRef &body_solution, MatrixRef &link_conditions,
                                   double eta) const;
    Eigen::MatrixXd flow_ellipsoidal(MatrixRef &r_trg, VectorRef &body_solution, MatrixRef &link_conditions,
                                     double eta) const;
    Eigen::MatrixXd flow_deformable(MatrixRef &r_trg, VectorRef &body_solution, MatrixRef &link_conditions,
                                    double eta) const;
    void step(VectorRef &body_sol, double dt) const;
    void set_evaluator(const std::string &evaluator);

    /// @brief Update cache variables for each Body. @see Body::update_cache_variables
    void update_cache_variables(double eta) {
        for (auto &body : bodies)
            body->update_cache_variables(eta);
    }

    /// @brief Get copy of a given nucleation site
    ///
    /// @param[in] i_body index of relevant body
    /// @param[in] j_site index of nucleation site
    Eigen::Vector3d get_nucleation_site(int i_body, int j_site) const {
        return bodies[i_body]->nucleation_sites_.col(j_site);
    };

    /// @brief get reference to ith Body without inspecting internal BodyContainer::bodies
    ///
    /// @param[in] i index of body in container
    /// @return Reference to ith Body in bodies
    const Body &at(size_t i) const { return *bodies.at(i); };

    /// @brief return number of bodies relevant for local calculations
    size_t get_local_count() const { return (world_rank_ == 0) ? bodies.size() : 0; };
    /// @brief return number of bodies relevant for global calculations
    size_t get_global_count() const { return bodies.size(); };

    /// @brief Return total number of body nodes across all MPI ranks
    size_t get_global_node_count() const {
        size_t n_nodes = 0;
        for (const auto &body : bodies)
            n_nodes += body->n_nodes_;
        return n_nodes;
    }

    /// @brief Return total number of body nucleation sites across all MPI ranks
    size_t get_global_site_count() const {
        size_t n_sites = 0;
        for (const auto &body : bodies)
            n_sites += body->nucleation_sites_.cols();
        return n_sites;
    }

    /// @brief msgpack serialization routine
    MSGPACK_DEFINE(spherical_bodies, deformable_bodies, ellipsoidal_bodies);
};

#endif
