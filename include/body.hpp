#ifndef BODY_HPP
#define BODY_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include <kernels.hpp>
#include <params.hpp>
#include <toml.hpp>

/// Class for "small" bodies such as MTOCs
class Body {
  public:
    int n_nodes_; ///< Number of nodes representing the body surface

    Eigen::Vector3d position_;       ///< Instantaneous lab frame position of body, usually the centroid
    Eigen::Quaterniond orientation_; ///< Instantaneous orientation of body
    const Eigen::Quaterniond orientation_ref_ = {1.0, 0.0, 0.0, 0.0}; ///< Reference orientation of body
    Eigen::Vector3d velocity_;                                        ///<  Net instantaneous lab frame velocity of body
    Eigen::Vector3d angular_velocity_;         ///< Net instantaneous lab frame angular velocity of body
    Eigen::Matrix<double, 6, 1> force_torque_; ///< Net force+torque vector [fx,fy,fz,tx,ty,tz] about centroid
    Eigen::VectorXd RHS_;                      ///< Current 'right-hand-side' for matrix formulation of solver

    Eigen::MatrixXd ex_; ///< [ 3 x num_nodes ] Singularity subtraction vector along x
    Eigen::MatrixXd ey_; ///< [ 3 x num_nodes ] Singularity subtraction vector along y
    Eigen::MatrixXd ez_; ///< [ 3 x num_nodes ] Singularity subtraction vector along z

    Eigen::MatrixXd K_; ///< [ 3*num_nodes x 6 ] matrix that helps translate body info to nodes

    Eigen::MatrixXd node_positions_;     ///< [ 3 x n_nodes ] node positions in lab frame
    Eigen::MatrixXd node_positions_ref_; ///< [ 3 x n_nodes ] node positions in reference 'body' frame
    Eigen::MatrixXd node_normals_;       ///< [ 3 x n_nodes ] node normals in lab frame
    Eigen::MatrixXd node_normals_ref_;   ///< [ 3 x n_nodes ] node normals in reference 'body' frame
    Eigen::VectorXd node_weights_;       ///< [ n_nodes ] far field quadrature weights for nodes

    /// [ 3 x n_nucleation_sites ] nucleation site positions in reference 'body' frame
    Eigen::MatrixXd nucleation_sites_ref_;
    /// [ 3 x n_nucleation_sites ] nucleation site positions in lab frame
    Eigen::MatrixXd nucleation_sites_;

    Eigen::MatrixXd A_;                         ///< Matrix representation of body for solver
    Eigen::PartialPivLU<Eigen::MatrixXd> A_LU_; ///< LU decomposition of A_ for preconditioner

    Body(const toml::table *body_table, const Params &params);

    const Eigen::Vector3d &get_position() const { return position_; };
    void update_RHS(const Eigen::Ref<const Eigen::MatrixXd> v_on_body);
    void update_cache_variables(double eta);
    void update_K_matrix();
    void update_preconditioner(double eta);
    void update_singularity_subtraction_vecs(double eta);
    void load_precompute_data(const std::string &input_file);
    void move(const Eigen::Vector3d &new_pos, const Eigen::Quaterniond &new_orientation);

    // For structures with fixed size Eigen::Vector types, this ensures alignment if the
    // structure is allocated via `new`
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class BodyContainer {
  public:
    std::vector<Body> bodies;
    /// Empty container constructor to avoid initialization list complications. No way to
    /// initialize after using this constructor, so overwrite objects with full constructor.
    BodyContainer(){};
    BodyContainer(toml::array *body_tables, Params &params);

    /// Pointer to FMM stresslet kernel (Stokes)
    std::unique_ptr<kernels::FMM<stkfmm::Stk3DFMM>> stresslet_kernel_;
    /// Pointer to FMM oseen kernel (PVel)
    std::unique_ptr<kernels::FMM<stkfmm::Stk3DFMM>> oseen_kernel_;

    /// @brief Get total number of nodes associated with body
    ///
    /// See BodyContainer::get_local_solution_size for notes about how this is distributed
    int get_local_node_count() const {
        if (world_rank_ != 0)
            return 0;

        int tot = 0;
        for (const auto &body : bodies)
            tot += body.n_nodes_;
        return tot;
    }

    /// @brief Get the size of all local bodies contribution to the matrix problem solution
    ///
    /// Since there aren't many bodies, and there is no easy way to synchronize them across processes, the rank 0
    /// process handles all components of the solution.
    int get_local_solution_size() const {
        return (world_rank_ == 0) ? 3 * get_local_node_count() + 6 * bodies.size() : 0;
    }

    void update_RHS(const Eigen::Ref<const Eigen::MatrixXd> &v_on_body);
    Eigen::VectorXd get_RHS() const;
    Eigen::MatrixXd get_global_node_positions() const;
    Eigen::MatrixXd get_local_node_positions() const;
    Eigen::MatrixXd get_node_normals() const;
    Eigen::MatrixXd get_center_positions(bool override_distributed) const {
        if (world_rank_ != 0 && !override_distributed)
            return Eigen::MatrixXd(3, 0);

        Eigen::MatrixXd centers(3, bodies.size());
        for (int i = 0; i < bodies.size(); ++i) {
            centers.block(0, i, 3, 1) = bodies[i].position_;
        }

        return centers;
    };

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
    unpack_solution_vector(const Eigen::Ref<const Eigen::VectorXd> &x) const;

    Eigen::MatrixXd get_local_center_positions() const { return get_center_positions(false); };
    Eigen::MatrixXd get_global_center_positions() const { return get_center_positions(true); };
    Eigen::VectorXd matvec(const Eigen::Ref<const Eigen::MatrixXd> &v_bodies,
                           const Eigen::Ref<const Eigen::MatrixXd> &body_densities,
                           const Eigen::Ref<const Eigen::MatrixXd> &body_velocities) const;

    Eigen::MatrixXd flow(const Eigen::Ref<const Eigen::MatrixXd> &r_trg,
                         const Eigen::Ref<const Eigen::MatrixXd> &densities,
                         const Eigen::Ref<const Eigen::MatrixXd> &force_torque_bodies, double eta) const;

    void update_cache_variables(double eta) {
        for (auto &body : bodies)
            body.update_cache_variables(eta);
    }

    Eigen::Vector3d get_nucleation_site(int i_body, int j_site) const {
        return bodies[i_body].nucleation_sites_.col(j_site);
    };

    const Body &at(size_t i) const { return bodies.at(i); };

    /// @brief return number of bodies relevant for local calculations
    size_t get_local_count() const { return (world_rank_ == 0) ? bodies.size() : 0; };
    /// @brief return number of bodies relevant for global calculations
    size_t get_global_count() const { return bodies.size(); };

  private:
    int world_rank_ = 0;
    int world_size_;
};

#endif
