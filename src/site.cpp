#include <functional>

#include <rng.hpp>
#include <site.hpp>

#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

SiteContainer::SiteContainer(toml::value &site_group_table) {
    spdlog::info("Initializing SiteContainer");
    capture_radius_ = toml::find_or(site_group_table, "capture_radius", 0.5);
    k_on_ = toml::find_or(site_group_table, "k_on", 0.0);
    k_off_ = toml::find_or(site_group_table, "k_off", 0.0);

    toml::array site_tables = toml::find(site_group_table, "sites").as_array();
    for (toml::value &site_config : site_tables)
        insert(site_config);

    bound_.resize(this->size());
}

void SiteContainer::kmc_step(const double &dt) {
    using namespace RNG;
    const int n_activate = poisson_int_unsplit(n_inactive() * k_on_ * dt);
    const int n_deactivate = poisson_int_unsplit(n_active() * k_off_ * dt);

    for (int i = 0; i < n_activate; ++i)
        activate(uniform_int_unsplit(0, n_inactive()));

    for (int i = 0; i < n_deactivate; ++i)
        deactivate(uniform_int_unsplit(0, n_active()));
}

void SiteContainer::insert(const toml::value &site_config) {
    Eigen::Vector3d x = Eigen::Map<Eigen::Vector3d>(toml::find<std::vector<double>>(site_config, "x").data());

    std::size_t site_index = pos_.cols();
    pos_.conservativeResize(Eigen::NoChange, site_index + 1);
    pos_.col(site_index) = x;

    sublist *states[] = {&inactive_, &active_};
    int state = toml::find_or<int>(site_config, "state", 0);
    if (state > 1 || state < 0)
        throw std::runtime_error("Invalid site state \"" + std::to_string(state) + ".\n");

    states[state]->push_back(site_index);

    spdlog::debug("Inserted site {} with position ({}, {}, {}) and state {}.", site_index, x[0], x[1], x[2], state);
}
