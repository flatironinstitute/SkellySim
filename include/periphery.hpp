#ifndef PERIPHERY_HPP
#define PERIPHERY_HPP

#include <Eigen/Core>
#include <iostream>
#include <vector>

class Periphery {
  public:
    Periphery() {};
    Periphery(const std::string &precompute_file);

    Eigen::MatrixXd flow(const Eigen::Ref<const Eigen::MatrixXd> &trg, const Eigen::Ref<const Eigen::MatrixXd> &density,
                         double eta) const;
    void compute_RHS(const Eigen::Ref<const Eigen::MatrixXd> v_on_shell);

    Eigen::MatrixXd M_inv_;                        // Process local elements of inverse matrix
    Eigen::MatrixXd stresslet_plus_complementary_; // Process local elements of stresslet tensor
    Eigen::MatrixXd node_pos_;
    Eigen::MatrixXd node_normal_;
    Eigen::VectorXd quadrature_weights_;
    Eigen::VectorXd RHS_;

    Eigen::VectorXi node_counts_;
    Eigen::VectorXi node_displs_;
    Eigen::VectorXi quad_counts_;
    Eigen::VectorXi quad_displs_;
    Eigen::VectorXi row_counts_;
    Eigen::VectorXi row_displs_;
    int n_nodes_global_ = 0;
};

#endif
