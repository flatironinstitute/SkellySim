#ifndef PERIPHERY_HPP
#define PERIPHERY_HPP

#include <skelly_sim.hpp>

#include <iostream>
#include <vector>

#include <kernels.hpp>
#include <params.hpp>

class SphericalBody;
class DeformableBody;
class EllipsoidalBody;
class FiberFiniteDifference;

/// Class to represent the containing boundary of the simulated system
///
/// There should be only periphery per system. The periphery, which is composed of smaller
/// discretized nodes, is distributed across all MPI ranks.
class Periphery {
  public:
    Periphery() = default;
    Periphery(const toml::value &periphery_table, const Params &params);

    Eigen::MatrixXd flow(CMatrixRef &trg, CMatrixRef &density, double eta) const;

    /// @brief Get the number of nodes local to the MPI rank
    int get_local_node_count() const { return M_inv_.rows() / 3; };

    /// @brief Get the rank local size of shell's contribution to the matrix problem solution
    int get_local_solution_size() const { return M_inv_.rows(); };

    /// @brief Get the global size of the shell's contribution to the matrix problem solution
    int get_global_solution_size() const { return M_inv_.cols(); };

    Eigen::MatrixXd get_local_node_positions() const { return node_pos_; };

    void update_RHS(CMatrixRef &v_on_shell);

    Eigen::VectorXd get_RHS() const { return RHS_; };

    Eigen::VectorXd apply_preconditioner(CVectorRef &x) const;
    Eigen::VectorXd matvec(CVectorRef &x_local, CMatrixRef &v_local) const;

    /// pointer to FMM object (pointer to avoid constructing object with empty Periphery)
    kernels::Evaluator stresslet_kernel_;
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

    void step(CVectorRef &solution) { solution_vec_ = solution; }
    void set_evaluator(const std::string &evaluator);

    bool is_active() const { return n_nodes_global_; }

    virtual bool check_collision(const DeformableBody &body, double threshold) const {
        if (!n_nodes_global_)
            return false;
        // FIXME: there is probably a way to make our objects abstract base classes, but it makes the containers weep if
        // you make this a pure virtual function, so instead we just throw an error.
        throw std::runtime_error("Collision (DeformableBody) undefined on base Periphery class\n");
    };

    virtual bool check_collision(const SphericalBody &body, double threshold) const {
        if (!n_nodes_global_)
            return false;
        // FIXME: there is probably a way to make our objects abstract base classes, but it makes the containers weep if
        // you make this a pure virtual function, so instead we just throw an error.
        throw std::runtime_error("Collision (SphericalBody) undefined on base Periphery class\n");
    };

    virtual bool check_collision(const EllipsoidalBody &body, double threshold) const {
        if (!n_nodes_global_)
            return false;
        // FIXME: there is probably a way to make our objects abstract base classes, but it makes the containers weep if
        // you make this a pure virtual function, so instead we just throw an error.
        throw std::runtime_error("Collision (EllipsoidalBody) undefined on base Periphery class\n");
    };

    virtual bool check_collision(const CMatrixRef &point_cloud, double threshold) const {
        if (!n_nodes_global_)
            return false;
        throw std::runtime_error("Collision undefined on base Periphery class\n");
    };

    // FIXME XXX: Fix this to be more clever in terms of taking in and spitting out fibers (containers?)
    virtual Eigen::MatrixXd fiber_interaction(const FiberFiniteDifference &fiber,
                                              const fiber_periphery_interaction_t &fp_params) const {
        throw std::runtime_error("fiber_interaction_finitediff undefined on base Periphery class\n");
    }

    virtual std::tuple<double, double, double> get_dimensions() {
        if (!n_nodes_global_)
            return {0.0, 0.0, 0.0};
        throw std::runtime_error("Point cloud interaction undefined on base Periphery class\n");
    }

    int n_nodes_global_ = 0; ///< Number of nodes across ALL MPI ranks
#ifdef SKELLY_DEBUG
    MSGPACK_DEFINE_MAP(solution_vec_, RHS_);
#else
    MSGPACK_DEFINE_MAP(solution_vec_);
#endif

  protected:
    int world_size_;
    int world_rank_ = -1;
};

class SphericalPeriphery : public Periphery {
  public:
    double radius_;
    SphericalPeriphery(const toml::value &periphery_table, const Params &params) : Periphery(periphery_table, params) {
        radius_ = toml::find_or<double>(periphery_table, "radius", 0.0);
        spdlog::info("  Spherical periphery radius: {}", radius_);
    };

    virtual bool check_collision(const SphericalBody &body, double threshold) const;
    virtual bool check_collision(const DeformableBody &body, double threshold) const;
    virtual bool check_collision(const EllipsoidalBody &body, double threshold) const;
    virtual bool check_collision(const CMatrixRef &point_cloud, double threshold) const;
    virtual Eigen::MatrixXd fiber_interaction(const FiberFiniteDifference &fiber,
                                              const fiber_periphery_interaction_t &fp_params) const;
    virtual std::tuple<double, double, double> get_dimensions() { return {radius_, radius_, radius_}; };
};

class EllipsoidalPeriphery : public Periphery {
  public:
    double a_;
    double b_;
    double c_;
    EllipsoidalPeriphery(const toml::value &periphery_table, const Params &params)
        : Periphery(periphery_table, params) {
        a_ = toml::find_or<double>(periphery_table, "a", 0.0);
        b_ = toml::find_or<double>(periphery_table, "b", 0.0);
        c_ = toml::find_or<double>(periphery_table, "c", 0.0);
        spdlog::info("  Ellipsoidal periphery a,b,c: {}, {}, {}", a_, b_, c_);
    };

    virtual bool check_collision(const SphericalBody &body, double threshold) const;
    virtual bool check_collision(const DeformableBody &body, double threshold) const;
    virtual bool check_collision(const EllipsoidalBody &body, double threshold) const;
    virtual bool check_collision(const CMatrixRef &point_cloud, double threshold) const;
    virtual Eigen::MatrixXd fiber_interaction(const FiberFiniteDifference &fiber,
                                              const fiber_periphery_interaction_t &fp_params) const;
    virtual std::tuple<double, double, double> get_dimensions() { return {a_, b_, c_}; };
};

class GenericPeriphery : public Periphery {
  public:
    double a_;
    double b_;
    double c_;
    GenericPeriphery(const toml::value &periphery_table, const Params &params) : Periphery(periphery_table, params) {
        double a = node_pos_.row(0).array().abs().maxCoeff();
        double b = node_pos_.row(1).array().abs().maxCoeff();
        double c = node_pos_.row(2).array().abs().maxCoeff();
        MPI_Allreduce(&a, &a_, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&b, &b_, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&c, &c_, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    };

    virtual bool check_collision(const SphericalBody &body, double threshold) const;
    virtual bool check_collision(const DeformableBody &body, double threshold) const;
    virtual bool check_collision(const EllipsoidalBody &body, double threshold) const;
    virtual bool check_collision(const CMatrixRef &point_cloud, double threshold) const;
    virtual Eigen::MatrixXd fiber_interaction(const FiberFiniteDifference &fiber,
                                              const fiber_periphery_interaction_t &fp_params) const;
    virtual std::tuple<double, double, double> get_dimensions() { return {a_, b_, c_}; };
};

#endif
