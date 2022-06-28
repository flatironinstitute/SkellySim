#include <functional>

#include <fiber.hpp>
#include <rng.hpp>
#include <site.hpp>
#include <utils.hpp>

#include <stdexcept>
#include <string>

#include <mpi.h>
#include <spdlog/spdlog.h>

SiteContainer::SiteContainer(toml::value &site_group_table) {
    spdlog::info("Initializing SiteContainer");

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);

    capture_radius_ = toml::find_or(site_group_table, "capture_radius", 0.5);
    k_on_ = toml::find_or(site_group_table, "k_on", 0.0);
    k_off_ = toml::find_or(site_group_table, "k_off", 0.0);

    toml::array site_tables = toml::find(site_group_table, "sites").as_array();
    for (toml::value &site_config : site_tables)
        insert(site_config);

    attached_.resize(this->size());
}

void SiteContainer::sync_attachments() {
    auto global_attachment_queue = utils::allgatherv(attachment_queue_);
    for (const auto &el : global_attachment_queue)
        attach(el.first, el.second);
    attachment_queue_.clear();
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

void SiteContainer::attach(const std::size_t &site_id, const global_fiber_pointer &p) {
    attached_[site_id] = p;
    if (mpi_rank_ == p.rank)
        p.fib->attach_to_site(site_id, *this);

    for (int i = 0; i < detached_.size(); ++i) {
        if (detached_[i] == site_id) {
            detached_[i] = detached_.back();
            detached_.pop_back();
            return;
        }
    }
    spdlog::get("SkellySim global")
        ->error("SiteContainer::attach: Unable to find detached site with global id {} to allow for attachment",
                site_id);
    throw std::runtime_error("site_id error");
}

void SiteContainer::detach(const std::size_t &site_id) {
    auto &site = attached_[site_id];
    if (site.rank == mpi_rank_)
        site.fib->detach_from_site(site_id);

    attached_[site_id] = {.rank = 0, .fib = nullptr};
    detached_.push_back(site_id);
}

void SiteContainer::activate(const std::size_t &inactive_index) {
    swap_state(inactive_index, inactive_, active_);
    detached_.push_back(active_.back());
}

void SiteContainer::deactivate(const std::size_t &active_index) {
    const int site = active_[active_index];
    attached_[site] = {.rank = 0, .fib = nullptr};
    swap_state(active_index, active_, inactive_);
    for (int i = 0; i < detached_.size(); ++i) {
        if (detached_[i] == site) {
            detached_[i] = detached_.back();
            detached_.pop_back();
            break;
        }
    }
}

void SiteContainer::insert(const toml::value &site_config) {
    Eigen::Vector3d x = Eigen::Map<Eigen::Vector3d>(toml::find<std::vector<double>>(site_config, "x").data());

    std::size_t site_index = pos_.cols();
    pos_.conservativeResize(Eigen::NoChange, site_index + 1);
    pos_.col(site_index) = x;

    inactive_.push_back(site_index);
    int state = toml::find_or<int>(site_config, "state", 0);
    if (state > 1 || state < 0)
        throw std::runtime_error("Invalid site state \"" + std::to_string(state) + ".\n");

    if (state == 1)
        activate(inactive_.size() - 1);

    spdlog::debug("Inserted site {} with position ({}, {}, {}) and state {}.", site_index, x[0], x[1], x[2], state);
}
