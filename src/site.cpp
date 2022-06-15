#include <functional>
#include <site.hpp>

#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

SiteContainer::SiteContainer(toml::array &site_tables) {
    spdlog::info("Initializing SiteContainer");
    for (toml::value &site_config : site_tables)
        insert(site_config);
}

void SiteContainer::insert(const toml::value &site_config) {
    Eigen::Vector3d x = Eigen::Map<Eigen::Vector3d>(toml::find<std::vector<double>>(site_config, "x").data());

    std::size_t site_index = pos_.cols();
    pos_.conservativeResize(Eigen::NoChange, site_index + 1);
    pos_.col(site_index) = x;

    std::reference_wrapper<sublist> states[] = {inactive_, active_, bound_};
    int state = toml::find_or<int>(site_config, "state", 0);
    if (state > 2 || state < 0)
        throw std::runtime_error("Invalid site state \"" + std::to_string(state) + ".\n");

    states[state].get().push_back(site_index);

    spdlog::debug("Inserted site {} with position ({}, {}, {}) and state {}.", site_index, x[0], x[1], x[2], state);
}
