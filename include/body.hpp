#ifndef BODY_HPP
#define BODY_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/LU>

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

    Eigen::MatrixXd A_;                         ///< Matrix representation of body for solver
    Eigen::PartialPivLU<Eigen::MatrixXd> A_LU_; ///< LU decomposition of A_ for preconditioner

    Body(const toml::table *body_table, const Params &params);

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

    int get_local_total_body_nodes() const {
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
        return (world_rank_ == 0) ? get_local_total_body_nodes() + 6 * bodies.size() : 0;
    }

    Eigen::VectorXd get_RHS() const;

    void update_cache_variables(double eta) {
        for (auto &body : bodies)
            body.update_cache_variables(eta);
    }

    /// @brief Synchronize body objects across processes
    ///
    /// Forces/torques/positions/orientations must be synchronized since each rank's contribution to force/torque will
    /// be different. This routine will appropriately make all body objects across processes the same.
    void synchronize();

  private:
    int world_rank_ = 0;
    int world_size_;
};

#endif
