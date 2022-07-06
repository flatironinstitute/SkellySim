#include <functional>

#include <fiber.hpp>
#include <link.hpp>
#include <rng.hpp>
#include <utils.hpp>

#include <stdexcept>
#include <string>

#include <mpi.h>
#include <spdlog/spdlog.h>

LinkContainer::LinkContainer(toml::value &link_group_table) {
    spdlog::info("Initializing LinkContainer");

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);

    capture_radius_ = toml::find_or(link_group_table, "capture_radius", 0.5);
    k_on_ = toml::find_or(link_group_table, "k_on", 0.0);
    k_off_ = toml::find_or(link_group_table, "k_off", 0.0);

    toml::array link_tables = toml::find(link_group_table, "links").as_array();
    for (toml::value &link_config : link_tables)
        insert(link_config);

    attached_.resize(this->size());
}

void LinkContainer::sync_attachments(FiberContainer &fc) {
    auto global_attachment_queue = utils::allgatherv(attachment_queue_);
    for (const auto &el : global_attachment_queue)
        attach(el.first, el.second, fc);
    attachment_queue_.clear();
}

void LinkContainer::kmc_step(const double &dt) {
    using namespace RNG;
    const int n_activate = poisson_int_unsplit(n_inactive() * k_on_ * dt);
    const int n_deactivate = poisson_int_unsplit(n_active() * k_off_ * dt);

    for (int i = 0; i < n_activate; ++i)
        activate(uniform_int_unsplit(0, n_inactive()));

    for (int i = 0; i < n_deactivate; ++i)
        deactivate(uniform_int_unsplit(0, n_active()));
}

void LinkContainer::attach(const std::size_t &link_id, const global_fiber_pointer &p, FiberContainer &fc) {
    attached_[link_id] = p;
    if (mpi_rank_ == p.rank)
        fc.at(p.fib).attach_to_link(link_id, *this);

    for (int i = 0; i < detached_.size(); ++i) {
        if (detached_[i] == link_id) {
            detached_[i] = detached_.back();
            detached_.pop_back();
            return;
        }
    }
    spdlog::get("SkellySim global")
        ->error("LinkContainer::attach: Unable to find detached link with global id {} to allow for attachment",
                link_id);
    throw std::runtime_error("link_id error");
}

void LinkContainer::detach(const std::size_t &link_id, FiberContainer &fc) {
    auto &link = attached_[link_id];
    if (link.rank == mpi_rank_)
        fc.at(link.fib).detach_from_link(link_id);

    attached_[link_id] = {.rank = 0, .fib = -1};
    detached_.push_back(link_id);
}

void LinkContainer::activate(const std::size_t &inactive_index) {
    swap_state(inactive_index, inactive_, active_);
    detached_.push_back(active_.back());
}

void LinkContainer::deactivate(const std::size_t &active_index) {
    const int link = active_[active_index];
    attached_[link] = {.rank = 0, .fib = -1};
    swap_state(active_index, active_, inactive_);
    for (int i = 0; i < detached_.size(); ++i) {
        if (detached_[i] == link) {
            detached_[i] = detached_.back();
            detached_.pop_back();
            break;
        }
    }
}

void LinkContainer::insert(const toml::value &link_config) {
    Eigen::Vector3d x = Eigen::Map<Eigen::Vector3d>(toml::find<std::vector<double>>(link_config, "x").data());

    std::size_t link_index = pos_.cols();
    pos_.conservativeResize(Eigen::NoChange, link_index + 1);
    pos_.col(link_index) = x;

    inactive_.push_back(link_index);
    int state = toml::find_or<int>(link_config, "state", 0);
    if (state > 1 || state < 0)
        throw std::runtime_error("Invalid link state \"" + std::to_string(state) + ".\n");

    if (state == 1)
        activate(inactive_.size() - 1);

    spdlog::debug("Inserted link {} with position ({}, {}, {}) and state {}.", link_index, x[0], x[1], x[2], state);
}
