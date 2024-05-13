#ifndef POINT_SOURCE_HPP
#define POINT_SOURCE_HPP

#include <skelly_sim.hpp>
#include <vector>

class PointSource {
  public:
    double time_to_live_ = 0.0;
    Eigen::Vector3d position_ = {0.0, 0.0, 0.0};
    Eigen::Vector3d force_ = {0.0, 0.0, 0.0};
    Eigen::Vector3d torque_ = {0.0, 0.0, 0.0};

    PointSource(const toml::value &point_table);
};

class PointSourceContainer {
  public:
    PointSourceContainer() = default;
    PointSourceContainer(const toml::array &point_tables);

    Eigen::MatrixXd flow(CMatrixRef &r_trg, double eta, double time);
    std::vector<PointSource> points;
};

#endif
