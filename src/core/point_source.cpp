#include <kernels.hpp>
#include <parse_util.hpp>
#include <point_source.hpp>

PointSource::PointSource(const toml::value &point_table) {
    if (point_table.contains("position"))
        position_ = parse_util::convert_array<>(point_table.at("position").as_array());
    if (point_table.contains("force"))
        force_ = parse_util::convert_array<>(point_table.at("force").as_array());
    if (point_table.contains("torque"))
        torque_ = parse_util::convert_array<>(point_table.at("torque").as_array());
    if (point_table.contains("time_to_live"))
        time_to_live_ = point_table.at("time_to_live").as_floating();
}

Eigen::MatrixXd PointSourceContainer::flow(const CMatrixRef &r_trg, double eta, double time) {
    Eigen::MatrixXd vel = Eigen::MatrixXd::Zero(3, r_trg.cols());
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> forcers;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> torquers;

    for (auto &point : points) {
        // Check if deactivated. ttl of 0.0 implies always alive
        if (point.time_to_live_ && time >= point.time_to_live_)
            continue;
        if (point.force_.any())
            forcers.push_back(std::make_pair(point.position_, point.force_));
        if (point.torque_.any())
            torquers.push_back(std::make_pair(point.position_, point.torque_));
    }

    if (forcers.size()) {
        const int n_forcers = forcers.size();
        Eigen::MatrixXd positions(3, n_forcers);
        Eigen::MatrixXd forces(3, n_forcers);
        for (int i = 0; i < n_forcers; ++i) {
            positions.col(i) = forcers[i].first;
            forces.col(i) = forcers[i].second;
        }
        vel += kernels::oseen_tensor_contract_direct(positions, r_trg, forces, eta);
    }

    if (torquers.size()) {
        const int n_torquers = torquers.size();
        Eigen::MatrixXd positions(3, n_torquers);
        Eigen::MatrixXd forces(3, n_torquers);
        for (int i = 0; i < n_torquers; ++i) {
            positions.col(i) = torquers[i].first;
            forces.col(i) = torquers[i].second;
        }
        vel += kernels::rotlet(positions, r_trg, forces, eta);
    }

    return vel;
}

PointSourceContainer::PointSourceContainer(const toml::array &point_tables) {
    const int n_points_tot = point_tables.size();
    spdlog::info("Reading in {} points", n_points_tot);

    for (int i_point = 0; i_point < n_points_tot; ++i_point) {
        const toml::value &point_table = point_tables.at(i_point);
        points.emplace_back(PointSource(point_table));

        auto &point = points.back();
        auto position = point.position_;
        spdlog::info("Point {}: [ {}, {}, {} ]", i_point, position[0], position[1], position[2]);
    }
}
