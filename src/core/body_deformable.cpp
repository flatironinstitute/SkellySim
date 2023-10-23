#include <skelly_sim.hpp>

#include <body.hpp>
#include <body_deformable.hpp>
#include <body_spherical.hpp>
#include <cnpy.hpp>
#include <kernels.hpp>
#include <parse_util.hpp>
#include <periphery.hpp>
#include <utils.hpp>

void DeformableBody::min_copy(const std::shared_ptr<DeformableBody> &other) {}
void DeformableBody::update_RHS(MatrixRef &v_on_body) {}
void DeformableBody::update_cache_variables(double eta) {}
void DeformableBody::update_preconditioner(double eta) {}
void DeformableBody::load_precompute_data(const std::string &input_file) {}
void DeformableBody::step(double dt, VectorRef &body_solution) {}
Eigen::VectorXd DeformableBody::matvec(MatrixRef &v_bodies, VectorRef &body_solution) const {
    return Eigen::VectorXd();
}
Eigen::VectorXd DeformableBody::apply_preconditioner(VectorRef &x) const { return Eigen::VectorXd(); }
Eigen::Vector3d DeformableBody::get_position() const { return Eigen::Vector3d(); }

bool DeformableBody::check_collision(const Periphery &periphery, double threshold) const {
    return periphery.check_collision(*this, threshold);
}
bool DeformableBody::check_collision(const Body &body, double threshold) const {
    return body.check_collision(*this, threshold);
}
bool DeformableBody::check_collision(const SphericalBody &body, double threshold) const {
    return body.check_collision(*this, threshold);
}
bool DeformableBody::check_collision(const DeformableBody &body, double threshold) const {
    spdlog::warn("check_collision not defined for DeformableBody->DeformableBody");
    return false;
}
