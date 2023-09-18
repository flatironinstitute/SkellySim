#include <skelly_sim.hpp>

#include <fiber_container_finitedifference.hpp>
#include <system.hpp>
#include <utils.hpp>

/// @brief Constructor
FiberContainerFinitedifference::FiberContainerFinitedifference(toml::array &fiber_tables, Params &params)
    : FiberContainerBase(fiber_tables, params) {

    spdlog::debug("FiberContainerFinitedifference::FiberContainerFinitedifference");

    fiber_type_ = FIBERTYPE::FiniteDifference;
    init_fiber_container(fiber_tables, params);

    spdlog::debug("FiberContainerFinitedifference::FiberContainerFinitedifference return");
}

/// @brief initialization function (overridden from base class)
///
/// This is a two-step method of initialization that is somewhat replicated, as the base class shouldn't call virtual
/// functions in the inherted classes, even though they look similar.
void FiberContainerFinitedifference::init_fiber_container(toml::array &fiber_tables, Params &params) {
    spdlog::debug("FiberContainerFinitedifference::init_fiber_container");

    const int n_fibs_tot = fiber_tables.size();
    const int n_fibs_extra = n_fibs_tot % world_size_;
    spdlog::info("Reading in {} fibers (finite difference).", n_fibs_tot);

    std::vector<int> displs(world_size_ + 1);
    for (int i = 1; i < world_size_ + 1; ++i) {
        displs[i] = displs[i - 1] + n_fibs_tot / world_size_;
        if (i <= n_fibs_extra)
            displs[i]++;
    }

    for (int i_fib = 0; i_fib < n_fibs_tot; ++i_fib) {
        const int i_fib_low = displs[world_rank_];
        const int i_fib_high = displs[world_rank_ + 1];

        if (i_fib >= i_fib_low && i_fib < i_fib_high) {
            toml::value &fiber_table = fiber_tables.at(i_fib);
            fibers_.emplace_back(fiber_table, params.eta);

            auto &fib = fibers_.back();
            spdlog::get("SkellySim global")
                ->debug("FiberFiniteDifference {}: {} {} {}", i_fib, fib.n_nodes_, fib.bending_rigidity_, fib.length_);
        }
    }

    // Set the local nubmer of fibers, node count, and solution size
    int node_tot = 0;
    for (auto &fib : fibers_) {
        node_tot += fib.n_nodes_;
    }
    set_local_fiber_numbers(fibers_.size(), node_tot, node_tot * 4);

    // Update the node positions
    update_local_node_positions();

    spdlog::debug("FiberContainerFinitedifference::init_fiber_container return");
}

/// @brief update the local node positions
///
/// Updating the local node positions should always be something that we do, as all implementations have some semblance
/// of nodes for now.
void FiberContainerFinitedifference::update_local_node_positions() {
    r_fib_local_.resize(3, get_local_node_count());
    size_t offset = 0;
    for (const auto &fib : *this) {
        for (int i_pt = 0; i_pt < fib.n_nodes_; ++i_pt) {
            for (int i = 0; i < 3; ++i) {
                r_fib_local_(i, i_pt + offset) = fib.x_(i, i_pt);
            }
        }
        offset += fib.n_nodes_;
    }
}

