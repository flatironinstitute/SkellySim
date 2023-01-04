#include <background_source.hpp>

#include <parse_util.hpp>

BackgroundSource::BackgroundSource(const toml::value &background_table) {
    if (background_table.contains("components"))
        components_ = parse_util::convert_array<Eigen::Vector3i>(background_table.at("components").as_array());
    if (background_table.contains("scale_factor"))
        scale_factor_ = parse_util::convert_array<>(background_table.at("scale_factor").as_array());
    if (background_table.contains("uniform"))
        uniform_ = parse_util::convert_array<>(background_table.at("uniform").as_array());
}

Eigen::MatrixXd BackgroundSource::flow(const MatrixRef &r_trg, double eta) {
    Eigen::MatrixXd vel = Eigen::MatrixXd::Zero(3, r_trg.cols());

    for (int i = 0; i < r_trg.cols(); ++i)
        for (int j = 0; j < 3; ++j)
            vel(j, i) = uniform_[j] + r_trg(components_[j], i) * scale_factor_[j];

    return vel;
}
