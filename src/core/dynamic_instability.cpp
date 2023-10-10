#include <skelly_sim.hpp>

#include <numeric>

#include <body.hpp>
#include <fiber_container_base.hpp>
#include <fiber_container_finitedifference.hpp>
#include <fiber_finitedifference.hpp>
#include <params.hpp>
#include <rng.hpp>
#include <system.hpp>

namespace System {

/// @brief Nucleate/grow/destroy Fibers based on dynamic instability rules. See white paper for details
///
/// Modifies:
/// - FiberContainer::fibers [for nucleation/catastrophe]
/// - Fiber::v_growth_
/// - Fiber::length_
/// - Fiber::length_prev_
void dynamic_instability() {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    const double dt = System::get_properties().dt;
    FiberContainerBase *fc = System::get_fiber_container();
    BodyContainer &bc = *System::get_body_container();
    Params &params = *System::get_params();
    if (params.dynamic_instability.n_nodes == 0)
        return;

    std::vector<int> body_offsets(bc.bodies.size() + 1);
    int i_body = 0;
    // Build basically a cumulative distribution function of the number of binding sites over all the bodies
    // This allows for fast mapping of (body_index, site_index) <-> site_index_flat
    for (const auto &body : bc.bodies) {
        body_offsets[i_body + 1] = body_offsets[i_body] + body->nucleation_sites_.cols();
        i_body++;
    }

    // Return a flat index given a (body_index, site_index) pair
    auto site_index = [&body_offsets](std::pair<int, int> binding_site) -> int {
        assert(binding_site.first >= 0 && binding_site.second >= 0);
        return body_offsets[binding_site.first] + binding_site.second;
    };

    // Return a (body_index, site_index) pair given a flat index
    auto binding_site_from_index = [&body_offsets](int index) -> std::pair<int, int> {
        for (size_t i_body = 0; i_body < body_offsets.size() - 1; ++i_body) {
            if (index < body_offsets[i_body + 1]) {
                return {i_body, index - body_offsets[i_body]};
            }
        }
        return {-1, -1};
    };

    // Array of occupied sites across all bodies
    std::vector<uint8_t> occupied_flat(bc.get_global_site_count(), 0);

    // Get the fibers
    int n_fib_old = fc->get_global_fiber_number();
    int n_active_old_local = 0;
    // FIXME This is specified for finite difference fibers only
    if (fc->fiber_type_ == FiberContainerBase::FIBERTYPE::FiniteDifference) {
        FiberContainerFinitedifference *fibers_fd = static_cast<FiberContainerFinitedifference *>(fc);
        auto fib = fibers_fd->fibers_.begin();
        while (fib != fibers_fd->fibers_.end()) {
            fib->v_growth_ = params.dynamic_instability.v_growth;
            double f_cat = params.dynamic_instability.f_catastrophe;
            if (fib->is_plus_pinned()) {
                fib->v_growth_ *= params.dynamic_instability.v_grow_collision_scale;
                f_cat *= params.dynamic_instability.f_catastrophe_collision_scale;
            }
            if (fib->attached_to_body())
                n_active_old_local++;

            // Remove fiber if catastrophe event
            if (RNG::uniform() > exp(-dt * f_cat)) {
                fib = fibers_fd->fibers_.erase(fib);
            } else {
                if (fib->attached_to_body())
                    occupied_flat[site_index(fib->binding_site_)] = 1;
                fib->length_prev_ = fib->length_;
                fib->length_ += dt * fib->v_growth_;
                fib++;
            }
        }
    } else {
        throw std::runtime_error("dynamic_instability (grow_remove) fiber type " + std::to_string(fc->fiber_type_) +
                                 " not implemented");
    }
    int n_active_old = 0;
    MPI_Reduce(&n_active_old_local, &n_active_old, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Let rank 0 know which sites are occupied
    MPI_Reduce(mpi_rank == 0 ? MPI_IN_PLACE : occupied_flat.data(), occupied_flat.data(), occupied_flat.size(),
               MPI_BYTE, MPI_LOR, 0, MPI_COMM_WORLD);

    // Map of empty sites. inactive_sites[flat_index] = true. Use map for fast deletion when site is filled
    std::unordered_map<int, bool> inactive_sites;
    // Vector of (body_index, site_index) pairs that will have a new fiber attached to them
    std::vector<std::pair<int, int>> to_nucleate;
    if (mpi_rank == 0) {
        for (size_t i = 0; i < occupied_flat.size(); ++i)
            if (!occupied_flat[i])
                inactive_sites[i] = true;

        int n_inactive_old = occupied_flat.size() - n_active_old;
        int n_to_nucleate = std::min(RNG::poisson_int(dt * params.dynamic_instability.nucleation_rate * n_inactive_old),
                                     static_cast<int>(inactive_sites.size()));

        while (n_to_nucleate) {
            int passive_site_index =
                std::next(inactive_sites.begin(), RNG::uniform_int(0, inactive_sites.size()))->first;

            auto [i_body, i_site] = binding_site_from_index(passive_site_index);
            inactive_sites.erase(passive_site_index);
            to_nucleate.push_back({i_body, i_site});
            n_to_nucleate--;
        }
    }

    // Total number of (current) fibers on this rank
    int n_fibers = 0;
    // FIXME Again, do this specifically for the finite difference fibers
    if (fc->fiber_type_ == FiberContainerBase::FIBERTYPE::FiniteDifference) {
        const FiberContainerFinitedifference *fibers_fd = static_cast<const FiberContainerFinitedifference *>(fc);
        n_fibers = fibers_fd->fibers_.size();
    } else {
        throw std::runtime_error("dynamic_instability (n_fibers) fiber type " + std::to_string(fc->fiber_type_) +
                                 " not implemented");
    }
    // Total number of (current) fibers across ranks
    std::vector<int> fiber_counts(mpi_rank == 0 ? mpi_size : 0);
    MPI_Gather(&n_fibers, 1, MPI_INT, fiber_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    spdlog::info("Deleted {} fibers", n_fib_old - std::accumulate(fiber_counts.begin(), fiber_counts.end(), 0));

    // Structure to communicate new fibers to their ranks
    using fiber_struct = struct {
        int rank;
        std::pair<int, int> binding_site;
    };

    // List of fibers to nucleate with their rank tagged
    std::vector<fiber_struct> new_fibers;
    // Find ranks with fewest fibers and fill them first, to balance load
    for (const auto &binding_site : to_nucleate) {
        const int fiber_rank = std::min_element(fiber_counts.begin(), fiber_counts.end()) - fiber_counts.begin();
        new_fibers.push_back({fiber_rank, binding_site});
        fiber_counts[fiber_rank]++;
    }

    int n_new = new_fibers.size();
    // Let all MPI ranks know about the fibers to be nucleated
    MPI_Bcast(&n_new, 1, MPI_INT, 0, MPI_COMM_WORLD);
    new_fibers.resize(n_new);
    MPI_Bcast(new_fibers.data(), sizeof(fiber_struct) * new_fibers.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
    if (n_new)
        spdlog::info("Sent {} fibers to nucleate", new_fibers.size());

    // Nucleate the fibers
    for (const auto &min_fib : new_fibers) {
        if (min_fib.rank == mpi_rank) {
            // FIXME finite difference fibers only for now
            if (fc->fiber_type_ == FiberContainerBase::FIBERTYPE::FiniteDifference) {
                FiberContainerFinitedifference *fibers_fd = static_cast<FiberContainerFinitedifference *>(fc);
                FiberFiniteDifference fib(params.dynamic_instability.n_nodes, params.dynamic_instability.radius,
                                          params.dynamic_instability.min_length,
                                          params.dynamic_instability.bending_rigidity, params.eta);
                fib.v_growth_ = 0.0;
                fib.binding_site_ = min_fib.binding_site;

                Eigen::MatrixXd x(3, fib.n_nodes_);
                Eigen::ArrayXd s = Eigen::ArrayXd::LinSpaced(fib.n_nodes_, 0, fib.length_).transpose();
                Eigen::Vector3d origin = bc.get_nucleation_site(fib.binding_site_.first, fib.binding_site_.second);
                Eigen::Vector3d u = (origin - bc.bodies[fib.binding_site_.first]->get_position()).normalized();

                for (int i = 0; i < 3; ++i)
                    fib.x_.row(i) = origin(i) + u(i) * s;

                fibers_fd->fibers_.push_back(fib);
                spdlog::get("SkellySim global")
                    ->debug("Inserted fiber on rank {} at site [{}, {}]", mpi_rank, min_fib.binding_site.first,
                            min_fib.binding_site.second);
            }
        }
    }
    spdlog::info("Nucleated {} fibers", new_fibers.size());

    // Each rank needs to go make it's internal numbers for fiber number, local node number, and local solution size
    if (fc->fiber_type_ == FiberContainerBase::FIBERTYPE::FiniteDifference) {
        FiberContainerFinitedifference *fibers_fd = static_cast<FiberContainerFinitedifference *>(fc);
        int node_tot = 0;
        for (auto &fib : fibers_fd->fibers_) {
            node_tot += fib.n_nodes_;
        }
        fibers_fd->set_local_fiber_numbers(fibers_fd->fibers_.size(), node_tot, node_tot * 4);
    }
}

} // namespace System
