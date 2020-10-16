#ifndef KERNELS_HPP
#define KERNELS_HPP
#include <Eigen/Dense>

namespace kernels {
Eigen::MatrixXd oseen_tensor_contract_direct(const Eigen::Ref<Eigen::MatrixXd> &r_src,
                                             const Eigen::Ref<Eigen::MatrixXd> &r_trg,
                                             const Eigen::Ref<Eigen::MatrixXd> &density, double eta = 1.0,
                                             double reg = 5E-3, double epsilon_distance = 1E-5);

Eigen::MatrixXd oseen_tensor_contract_fmm(const Eigen::Ref<Eigen::MatrixXd> &r_src,
                                          const Eigen::Ref<Eigen::MatrixXd> &r_trg,
                                          const Eigen::Ref<Eigen::MatrixXd> &f_src);

Eigen::MatrixXd oseen_tensor_direct(const Eigen::Ref<Eigen::MatrixXd> &r_src, const Eigen::Ref<Eigen::MatrixXd> &r_trg,
                                    double eta = 1.0, double reg = 5E-3, double epsilon_distance = 1E-5);

}; // namespace kernels

#endif
