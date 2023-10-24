#ifndef BODY_DEFORMABLE_HPP
#define BODY_DEFORMABLE_HPP

#include <skelly_sim.hpp>

#include <Eigen/LU>
#include <body.hpp>
#include <kernels.hpp>
#include <params.hpp>

class Periphery;
class SphericalBody;
class EllipsoidalBody;
class FiberContainerFiniteDifference;

/// @brief Spherical Body...
class DeformableBody : public Body {
  public:
    /// @brief Construct deformable body. @see Body
    /// @param[in] body_table Parsed TOML body table
    /// @param[in] params Initialized Params object
    DeformableBody(const toml::value &body_table, const Params &params) : Body(body_table, params){};
    DeformableBody() = default;

    /// Duplicate SphericalBody object
    std::shared_ptr<Body> clone() const override { return std::make_shared<DeformableBody>(*this); };

    void min_copy(const std::shared_ptr<DeformableBody> &other);

    void update_RHS(MatrixRef &v_on_body) override;
    void update_cache_variables(double eta) override;
    void update_preconditioner(double eta) override;
    void load_precompute_data(const std::string &input_file) override;
    void step(double dt, VectorRef &body_solution) override;
    int get_solution_size() const override { return n_nodes_ * 4; };
    Eigen::VectorXd matvec(MatrixRef &v_bodies, VectorRef &body_solution) const override;
    Eigen::VectorXd apply_preconditioner(VectorRef &x) const override;
    Eigen::Vector3d get_position() const override;

    bool check_collision(const Periphery &periphery, double threshold) const override;
    bool check_collision(const Body &body, double threshold) const override;
    bool check_collision(const SphericalBody &body, double threshold) const override;
    bool check_collision(const DeformableBody &body, double threshold) const override;
    bool check_collision(const EllipsoidalBody &body, double threshold) const override;

    MSGPACK_DEFINE_MAP(node_positions_, node_normals_, solution_vec_);
};

#endif
