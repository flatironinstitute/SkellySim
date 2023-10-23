#ifndef BODY_SPHERICAL_HPP
#define BODY_SPHERICAL_HPP

#include <skelly_sim.hpp>

#include <Eigen/LU>
#include <body.hpp>
#include <kernels.hpp>
#include <params.hpp>

class Periphery;
class DeformableBody;
class FiberContainerFiniteDifference;

/// @brief Spherical Body...
class SphericalBody : public Body {
  public:
    /// @brief Construct spherical body. @see Body
    /// @param[in] body_table Parsed TOML body table. Must have 'radius' key defined.
    /// @param[in] params Initialized Params object
    SphericalBody(const toml::value &body_table, const Params &params);
    SphericalBody() = default;

    /// Duplicate SphericalBody object
    std::shared_ptr<Body> clone() const override { return std::make_shared<SphericalBody>(*this); };

    // Parameters unique to spherical body
    double radius_;                  ///< Radius of body
    Eigen::Vector3d position_;       ///< Instantaneous lab frame position of body, usually the centroid
    Eigen::Quaterniond orientation_; ///< Instantaneous orientation of body
    Eigen::Quaterniond orientation_ref_ = {1.0, 0.0, 0.0, 0.0}; ///< Reference orientation of body
    Eigen::Vector3d velocity_;                                  ///<  Net instantaneous lab frame velocity of body
    Eigen::Vector3d angular_velocity_;         ///< Net instantaneous lab frame angular velocity of body
    Eigen::Matrix<double, 6, 1> force_torque_; ///< Net force+torque vector [fx,fy,fz,tx,ty,tz] about centroid

    Eigen::MatrixXd ex_; ///< [ 3 x num_nodes ] Singularity subtraction vector along x
    Eigen::MatrixXd ey_; ///< [ 3 x num_nodes ] Singularity subtraction vector along y
    Eigen::MatrixXd ez_; ///< [ 3 x num_nodes ] Singularity subtraction vector along z
    Eigen::MatrixXd K_;  ///< [ 3*num_nodes x 6 ] matrix that helps translate body info to nodes
    Eigen::Vector3d external_force_{0.0, 0.0,
                                    0.0}; ///< [3] vector of external force on body in lab frame (can be oscillatory)
    Eigen::Vector3d external_torque_{0.0, 0.0, 0.0}; ///< [3] vector of constant external torque on body in lab frame

    // Parameters controlling the type of externally prescribed force
    EXTFORCE external_force_type_ = EXTFORCE::Linear; ///< External force type [Linear, Oscillatory]
    double extforce_oscillation_amplitude_ = 0.0;     ///< External force amplitude (if Oscillatory)
    double extforce_oscillation_omega_ = 0.0;         ///< External force angular frequency (if Oscillatory)
    double extforce_oscillation_phase_ = 0.0;         ///< External force phase shift (if Oscillatory)

    Eigen::MatrixXd A_;                         ///< Matrix representation of body for solver
    Eigen::PartialPivLU<Eigen::MatrixXd> A_LU_; ///< LU decomposition of A_ for preconditioner

    void update_RHS(MatrixRef &v_on_body) override;
    void update_cache_variables(double eta) override;
    void update_preconditioner(double eta) override;
    void load_precompute_data(const std::string &input_file) override;
    void step(double dt, VectorRef &body_solution) override;
    void min_copy(const std::shared_ptr<SphericalBody> &other);

    int get_solution_size() const override { return n_nodes_ * 3 + 6; };
    Eigen::VectorXd matvec(MatrixRef &v_bodies, VectorRef &body_solution) const override;
    Eigen::VectorXd apply_preconditioner(VectorRef &x) const override;

    Eigen::Vector3d get_position() const override { return position_; }
    void place(const Eigen::Vector3d &new_pos, const Eigen::Quaterniond &new_orientation);
    void update_K_matrix();
    void update_singularity_subtraction_vecs(double eta);

    bool check_collision(const Periphery &periphery, double threshold) const override;
    bool check_collision(const Body &body, double threshold) const override;
    bool check_collision(const SphericalBody &body, double threshold) const override;
    bool check_collision(const DeformableBody &body, double threshold) const override;

    /// @brief Serialize body automatically with msgpack macros
    MSGPACK_DEFINE_MAP(radius_, position_, orientation_, solution_vec_);
};

#endif
