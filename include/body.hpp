#ifndef BODY_HPP
#define BODY_HPP

#include <skelly_sim.hpp>

#include <Eigen/LU>
#include <kernels.hpp>
#include <params.hpp>

class Periphery;
class SphericalBody;
class DeformableBody;
class EllipsoidalBody;
class FiberContainerFiniteDifference;

/// Class for "small" bodies such as MTOCs
class Body {
  public:
    enum EXTFORCE { Linear, Oscillatory };    ///< Type of external force [Linear, Oscillatory]
    static const std::string EXTFORCE_name[]; ///< String name of external force
    int n_nodes_;                             ///< Number of nodes representing the body surface

    Eigen::VectorXd RHS_;                ///< Current 'right-hand-side' for matrix formulation of solver
    Eigen::MatrixXd node_positions_;     ///< [ 3 x n_nodes ] node positions in lab frame
    Eigen::MatrixXd node_positions_ref_; ///< [ 3 x n_nodes ] node positions in reference 'body' frame
    Eigen::MatrixXd node_normals_;       ///< [ 3 x n_nodes ] node normals in lab frame
    Eigen::MatrixXd node_normals_ref_;   ///< [ 3 x n_nodes ] node normals in reference 'body' frame
    Eigen::VectorXd node_weights_;       ///< [ n_nodes ] far field quadrature weights for nodes
    Eigen::VectorXd solution_vec_;       ///< [ 3 * n_nodes + <body_specific> ] strength of interaction on nodes

    /// [ 3 x n_nucleation_sites ] nucleation site positions in reference 'body' frame
    Eigen::MatrixXd nucleation_sites_ref_;
    /// [ 3 x n_nucleation_sites ] nucleation site positions in lab frame
    Eigen::MatrixXd nucleation_sites_;

    Body(const toml::value &body_table, const Params &params);
    Body() = default; ///< default constructor...

    virtual void update_RHS(CMatrixRef &v_on_body) = 0;
    virtual void update_cache_variables(double eta) = 0;
    virtual void update_preconditioner(double eta) = 0;
    virtual void load_precompute_data(const std::string &input_file) = 0;
    virtual void step(double dt, CVectorRef &body_solution) = 0;

    virtual int get_solution_size() const = 0;
    virtual Eigen::Vector3d get_position() const = 0;
    virtual Eigen::VectorXd matvec(CMatrixRef &v_bodies, CVectorRef &body_solution) const = 0;
    virtual Eigen::VectorXd apply_preconditioner(CVectorRef &x) const = 0;

    /// @brief Make a copy of this instance
    virtual std::shared_ptr<Body> clone() const = 0;

    /// @brief dummy method to be overriden by derived classes
    virtual bool check_collision(const Periphery &periphery, double threshold) const = 0;

    /// @brief dummy method to be overriden by derived classes
    virtual bool check_collision(const Body &body, double threshold) const = 0;

    /// @brief dummy method to be overriden by derived classes
    virtual bool check_collision(const SphericalBody &body, double threshold) const = 0;

    /// @brief dummy method to be overriden by derived classes
    virtual bool check_collision(const DeformableBody &body, double threshold) const = 0;

    /// @brief dummy method to be overriden by derived classes
    virtual bool check_collision(const EllipsoidalBody &body, double threshold) const = 0;

    /// For structures with fixed size Eigen::Vector types, this ensures alignment if the
    /// structure is allocated via `new`
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif
