#ifndef PERIPHERY_HPP
#define PERIPHERY_HPP

#include <iostream>
#include <Eigen/Core>

class Periphery {
  public:
    Periphery(const std::string &precompute_file);
    Eigen::MatrixXd M_inv_;
    Eigen::MatrixXd stresslet_plus_complementary_;
};

#endif
