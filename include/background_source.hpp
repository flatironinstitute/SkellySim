#ifndef BACKGROUND_SOURCE_HPP
#define BACKGROUND_SOURCE_HPP

#include <skelly_sim.hpp>

class BackgroundSource {
  public:
    BackgroundSource() = default;
    BackgroundSource(const toml::value &background_table);
    Eigen::MatrixXd flow(const MatrixRef &r_trg, double eta);
    bool is_active() { return uniform_.norm() + scale_factor_.norm(); }

  private:
    Eigen::Vector3i components_ = {0, 1, 2};
    Eigen::Vector3d scale_factor_ = {0.0, 0.0, 0.0};
    Eigen::Vector3d uniform_ = {0.0, 0.0, 0.0};
};

#endif
