#ifndef PERIPHERY_HPP
#define PERIPHERY_HPP

#include <Eigen/Core>
#include <Tpetra_Core.hpp>

class Periphery {
  public:
    virtual Eigen::VectorXd h(const Eigen::Ref<Eigen::MatrixXd> &p) = 0;
    virtual Eigen::MatrixXd gradh(const Eigen::Ref<Eigen::MatrixXd> &p) = 0;

  protected:
    Eigen::MatrixXd quadrature_nodes_;
    Eigen::MatrixXd nodes_normal_;
};

class SphericalPeriphery : public Periphery {
  public:
    SphericalPeriphery(int n_nodes, double radius);
    Eigen::VectorXd h(const Eigen::Ref<Eigen::MatrixXd> &p);
    Eigen::MatrixXd gradh(const Eigen::Ref<Eigen::MatrixXd> &p);

  private:
    double radius_;
};

#endif
