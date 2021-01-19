#ifndef BODY_HPP
#define BODY_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include <params.hpp>
#include <toml.hpp>

class Body {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int num_nodes_;

    Eigen::Vector3d position_;
    Eigen::Quaterniond orientation_;
    const Eigen::Quaterniond orientation_ref_ = {1.0, 0.0, 0.0, 0.0};
    Eigen::Vector3d velocity_;
    Eigen::Vector3d angular_velocity_;
    Eigen::Matrix<double, 6, 1> force_torque_;
    Eigen::VectorXd RHS_;

    Eigen::MatrixXd ex_;
    Eigen::MatrixXd ey_;
    Eigen::MatrixXd ez_;

    Eigen::MatrixXd K_;

    Eigen::MatrixXd node_positions_; // absolute positions of nodes
    Eigen::MatrixXd node_positions_ref_; // node positions in reference frame
    Eigen::MatrixXd node_normals_;
    Eigen::MatrixXd node_normals_ref_; // node normals in reference frame
    Eigen::VectorXd node_weights_;

    Eigen::MatrixXd A_;
    Eigen::PartialPivLU<Eigen::MatrixXd> A_LU_;

    Body(const toml::table *body_table, const Params &params);

    void compute_RHS(const Eigen::Ref<const Eigen::MatrixXd> v_on_body);
    void update_cache(double eta);
    void update_K_matrix();
    void update_preconditioner(double eta);
    void build_preconditioner(double eta);
    void load_precompute_data(const std::string &input_file);
    void move(const Eigen::Vector3d &new_pos, const Eigen::Quaterniond &new_orientation);
    void update_singularity_subtraction_vecs(double eta);
};

class BodyContainer {};

#endif
