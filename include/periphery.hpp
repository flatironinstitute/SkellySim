#ifndef PERIPHERY_HPP
#define PERIPHERY_HPP

#include <skelly_sim.hpp>

#include <iostream>
#include <vector>

#include <kernels.hpp>
#include <params.hpp>

class SphericalBody;
class DeformableBody;
class Fiber;

/// Class to represent the containing boundary of the simulated system
///
/// There should be only periphery per system. The periphery, which is composed of smaller
/// discretized nodes, is distributed across all MPI ranks.
class Periphery {
  public:
    Periphery() = default;
    Periphery(const std::string &precompute_file, const toml::value &periphery_table, const Params &params);

    Eigen::MatrixXd flow(MatrixRef &trg, MatrixRef &density, double eta) const;

    /// @brief Get the number of nodes local to the MPI rank
    int get_local_node_count() const { return M_inv_.rows() / 3; };

    /// @brief Get the size of the shell's contribution to the matrix problem solution
    int get_local_solution_size() const { return M_inv_.rows(); };

    Eigen::MatrixXd get_local_node_positions() const { return node_pos_; };

    void update_RHS(MatrixRef &v_on_shell);

    Eigen::VectorXd get_RHS() const { return RHS_; };

    Eigen::VectorXd apply_preconditioner(VectorRef &x) const;
    Eigen::VectorXd matvec(VectorRef &x_local, MatrixRef &v_local) const;

    /// pointer to FMM object (pointer to avoid constructing object with empty Periphery)
    std::shared_ptr<kernels::FMM<stkfmm::Stk3DFMM>> stresslet_kernel_;
    Eigen::MatrixXd M_inv_;                            ///< Process local elements of inverse matrix
    Eigen::MatrixXd stresslet_plus_complementary_;     ///< Process local elements of stresslet tensor
    Eigen::MatrixXd node_pos_ = Eigen::MatrixXd(3, 0); ///< [3xn_nodes_local] matrix representing node positions
    Eigen::MatrixXd node_normal_;        ///< [3xn_nodes_local] matrix representing node normal vectors (inward facing)
    Eigen::VectorXd quadrature_weights_; ///< [n_nodes] array of 'far-field' quadrature weights
    Eigen::VectorXd RHS_;                ///< [ 3 * n_nodes ] Current 'right-hand-side' for matrix formulation of solver
    Eigen::VectorXd solution_vec_;       ///< [ 3 * n_nodes ] Current 'solution' for matrix formulation of solver

    /// MPI_WORLD_SIZE array that specifies node_counts_[i] = number_of_nodes_on_rank_i*3
    Eigen::VectorXi node_counts_;
    /// MPI_WORLD_SIZE+1 array that specifies node displacements. Is essentially the CDF of node_counts_
    Eigen::VectorXi node_displs_;
    /// MPI_WORLD_SIZE array that specifies quad_counts_[i] = number_of_nodes_on_rank_i
    Eigen::VectorXi quad_counts_;
    /// MPI_WORLD_SIZE+1 array that specifies quadrature displacements. Is essentially the CDF of quad_counts_
    Eigen::VectorXi quad_displs_;
    /// MPI_WORLD_SIZE array that specifies row_counts_[i] = 3 * n_nodes_global_ * number_of_nodes_on_rank_i
    Eigen::VectorXi row_counts_;
    /// MPI_WORLD_SIZE+1 array that specifies row displacements. Is essentially the CDF of row_counts_
    Eigen::VectorXi row_displs_;

    void step(VectorRef &solution) { solution_vec_ = solution; }

    virtual bool check_collision(const DeformableBody &body, double threshold) const {
        if (!n_nodes_global_)
            return false;
        // FIXME: there is probably a way to make our objects abstract base classes, but it makes the containers weep if
        // you make this a pure virtual function, so instead we just throw an error.
        throw std::runtime_error("Collision undefined on base Periphery class\n");
    };

    virtual bool check_collision(const SphericalBody &body, double threshold) const {
        if (!n_nodes_global_)
            return false;
        // FIXME: there is probably a way to make our objects abstract base classes, but it makes the containers weep if
        // you make this a pure virtual function, so instead we just throw an error.
        throw std::runtime_error("Collision undefined on base Periphery class\n");
    };

    virtual bool check_collision(const MatrixRef &point_cloud, double threshold) const {
        if (!n_nodes_global_)
            return false;
        throw std::runtime_error("Collision undefined on base Periphery class\n");
    };

    virtual Eigen::MatrixXd fiber_interaction(const Fiber &fiber,
                                              const fiber_periphery_interaction_t &fp_params) const {
        throw std::runtime_error("fiber_interaction undefined on base Periphery class\n");
    }

    virtual std::tuple<double, double, double> get_dimensions() {
        if (!n_nodes_global_)
            return {0.0, 0.0, 0.0};
        throw std::runtime_error("Point cloud interaction undefined on base Periphery class\n");
    }

    int n_nodes_global_ = 0; ///< Number of nodes across ALL MPI ranks
#ifdef SKELLY_DEBUG
    MSGPACK_DEFINE_MAP(RHS_);
#else
    MSGPACK_DEFINE_MAP(solution_vec_, RHS_);
#endif
    
  protected:
    int world_size_;
    int world_rank_ = -1;
};

class SphericalPeriphery : public Periphery {
  public:
    double radius_;
    SphericalPeriphery(const std::string &precompute_file, const toml::value &periphery_table, const Params &params)
        : Periphery(precompute_file, periphery_table, params) {
        radius_ = toml::find_or<double>(periphery_table, "radius", 0.0);
    };

    virtual bool check_collision(const SphericalBody &body, double threshold) const;
    virtual bool check_collision(const DeformableBody &body, double threshold) const;
    virtual bool check_collision(const MatrixRef &point_cloud, double threshold) const;
    virtual Eigen::MatrixXd fiber_interaction(const Fiber &fiber, const fiber_periphery_interaction_t &fp_params) const;
    virtual std::tuple<double, double, double> get_dimensions() { return {radius_, radius_, radius_}; };
};

class EllipsoidalPeriphery : public Periphery {
  public:
    double a_;
    double b_;
    double c_;
    EllipsoidalPeriphery(const std::string &precompute_file, const toml::value &periphery_table, const Params &params)
        : Periphery(precompute_file, periphery_table, params) {
        a_ = toml::find_or<double>(periphery_table, "a", 0.0);
        b_ = toml::find_or<double>(periphery_table, "b", 0.0);
        c_ = toml::find_or<double>(periphery_table, "c", 0.0);
    };

    virtual bool check_collision(const SphericalBody &body, double threshold) const;
    virtual bool check_collision(const DeformableBody &body, double threshold) const;
    virtual bool check_collision(const MatrixRef &point_cloud, double threshold) const;
    virtual Eigen::MatrixXd fiber_interaction(const Fiber &fiber, const fiber_periphery_interaction_t &fp_params) const;
    virtual std::tuple<double, double, double> get_dimensions() { return {a_, b_, c_}; };
};

class GenericPeriphery : public Periphery {
  public:
    double a_;
    double b_;
    double c_;
    GenericPeriphery(const std::string &precompute_file, const toml::value &periphery_table, const Params &params)
        : Periphery(precompute_file, periphery_table, params) {
        a_ = node_pos_.row(0).array().abs().maxCoeff();
        b_ = node_pos_.row(1).array().abs().maxCoeff();
        c_ = node_pos_.row(2).array().abs().maxCoeff();
    };

    virtual bool check_collision(const SphericalBody &body, double threshold) const;
    virtual bool check_collision(const DeformableBody &body, double threshold) const;
    virtual bool check_collision(const MatrixRef &point_cloud, double threshold) const;
    virtual Eigen::MatrixXd fiber_interaction(const Fiber &fiber, const fiber_periphery_interaction_t &fp_params) const;
    virtual std::tuple<double, double, double> get_dimensions() { return {a_, b_, c_}; };
};

#endif
