#ifndef PERIPHERY_HPP
#define PERIPHERY_HPP

#include <iostream>
#include <Eigen/Core>

class Periphery {
  public:
    Periphery(const std::string &precompute_file);
    Eigen::MatrixXd M_inv_; // Process local elements of inverse matrix
    Eigen::MatrixXd stresslet_plus_complementary_; // Process local elements of stresslet tensor
};

#endif
