#include <skelly_sim.hpp>

#include <body.hpp>
#include <fiber_container_finite_difference.hpp>
#include <periphery.hpp>
#include <system.hpp>
#include <utils.hpp>

using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

/// @brief Empty constructor
FiberContainerFiniteDifference::FiberContainerFiniteDifference() {
    spdlog::debug("FiberContainerFiniteDifference::FiberContainerFiniteDifference (empty)");

    // This just exists to set the fiber type
    fiber_type_ = FIBERTYPE::FiniteDifference;

    spdlog::debug("FiberContainerFiniteDifference::FiberContainerFiniteDifference (empty) return");
}

/// @brief Constructor
FiberContainerFiniteDifference::FiberContainerFiniteDifference(toml::array &fiber_tables, Params &params)
    : FiberContainerBase(fiber_tables, params) {

    spdlog::debug("FiberContainerFiniteDifference::FiberContainerFiniteDifference");

    fiber_type_ = FIBERTYPE::FiniteDifference;
    init_fiber_container(fiber_tables, params);

    spdlog::debug("FiberContainerFiniteDifference::FiberContainerFiniteDifference return");
}

/// @bried for collision of fiber with periphery within some threshold
bool FiberContainerFiniteDifference::check_collision(const Periphery &periphery, double threshold) const {
    bool collided = false;
    for (const auto &fiber : *this) {
        if (fiber.is_minus_clamped()) {
            if (!collided && periphery.check_collision(fiber.x_.block(0, 1, 3, fiber.n_nodes_ - 1), threshold)) {
                collided = true;
            }
        } else {
            if (!collided && periphery.check_collision(fiber.x_, threshold)) {
                collided = true;
            }
        }
    }

    return collided;
}

// @brief Calculate max error from active fibers
double FiberContainerFiniteDifference::fiber_error_local() const {
    double error = 0.0;
    for (const auto &fib : *this) {
        const auto &mats = fib.matrices_.at(fib.n_nodes_);
        const Eigen::MatrixXd xs = std::pow(2.0 / fib.length_, 1) * fib.x_ * mats.D_1_0;
        for (int i = 0; i < fib.n_nodes_; ++i)
            error = std::max(fabs(xs.col(i).norm() - 1.0), error);
    }

    return error;
}

/// @brief initialization function (overridden from base class)
///
/// This is a two-step method of initialization that is somewhat replicated, as the base class shouldn't call virtual
/// functions in the inherted classes, even though they look similar.
void FiberContainerFiniteDifference::init_fiber_container(toml::array &fiber_tables, Params &params) {
    spdlog::debug("FiberContainerFiniteDifference::init_fiber_container");

    const int n_fibs_tot = fiber_tables.size();
    const int n_fibs_extra = n_fibs_tot % world_size_;
    spdlog::info("  Reading in {} fibers (finite difference).", n_fibs_tot);

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

    // Update the node positions
    update_local_node_positions();

    spdlog::debug("FiberContainerFiniteDifference::init_fiber_container return");
}

/// @brief update the local node positions
///
/// Updating the local node positions should always be something that we do, as all implementations have some semblance
/// of nodes for now.
void FiberContainerFiniteDifference::update_local_node_positions() {
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

/// @brief update the cache variables for ourselves
void FiberContainerFiniteDifference::update_cache_variables(double dt, double eta) {
    for (auto &fib : *this) {
        fib.update_constants(eta);
        fib.update_derivatives();
        fib.update_stokeslet(eta);
        fib.update_linear_operator(dt, eta);
        fib.update_force_operator();
    }

    update_local_node_positions();
}

/// @brief Generate a constant force
MatrixXd FiberContainerFiniteDifference::generate_constant_force() const {
    const int n_fib_pts = get_local_node_count();
    MatrixXd f(3, n_fib_pts);
    size_t offset = 0;
    for (const auto &fib : *this) {
        f.block(0, offset, 3, fib.n_nodes_) = fib.force_scale_ * fib.xs_;
        offset += fib.n_nodes_;
    }
    return f;
}

/// @brief Fiber flow
MatrixXd FiberContainerFiniteDifference::flow(const MatrixRef &r_trg, const MatrixRef &fib_forces, double eta,
                                              bool subtract_self) const {
    spdlog::debug("FiberContainerFinitediffere::flow starting");

    const size_t n_src = fib_forces.cols();
    const size_t n_trg = r_trg.cols();
    if (!get_global_fiber_count())
        return Eigen::MatrixXd::Zero(3, n_trg);

    MatrixXd weighted_forces(3, n_src);
    MatrixXd r_src = get_local_node_positions();
    size_t offset = 0;

    for (const auto &fib : *this) {
        const ArrayXd &weights = 0.5 * fib.length_ * fib.matrices_.at(fib.n_nodes_).weights_0;

        for (int i_pt = 0; i_pt < fib.n_nodes_; ++i_pt)
            for (int i = 0; i < 3; ++i)
                weighted_forces(i, i_pt + offset) = weights(i_pt) * fib_forces(i, i_pt + offset);

        offset += fib.n_nodes_;
    }

    // All-to-all
    MatrixXd r_dl_dummy, f_dl_dummy;
    utils::LoggerRedirect redirect(std::cout);
    MatrixXd vel = stokeslet_kernel_(r_src, r_dl_dummy, r_trg, weighted_forces, f_dl_dummy, eta);
    redirect.flush(spdlog::level::debug, "STKFMM");

    // Subtract self term
    offset = 0;
    if (subtract_self) {
        for (const auto &fib : *this) {
            VectorMap wf_flat(weighted_forces.data() + offset * 3, fib.n_nodes_ * 3);
            VectorMap vel_flat(vel.data() + offset * 3, fib.n_nodes_ * 3);
            vel_flat -= fib.stokeslet_ * wf_flat;
            offset += fib.n_nodes_;
        }
    }

    spdlog::debug("FiberContainerFinitediffere::flow finished");
    return vel;
}

Eigen::VectorXd FiberContainerFiniteDifference::matvec(VectorRef &x_all, MatrixRef &v_fib,
                                                       MatrixRef &v_fib_boundary) const {
    VectorXd res = VectorXd::Zero(get_local_solution_size());

    size_t offset = 0;
    size_t offset_node = 0;
    int i_fib = 0;
    for (const auto &fib : *this) {
        const int np = fib.n_nodes_;
        res.segment(offset, 4 * np) =
            fib.matvec(x_all.segment(offset, 4 * np), v_fib.block(0, offset_node, 3, np), v_fib_boundary.col(i_fib));

        i_fib++;
        offset += 4 * np;
        offset_node += np;
    }

    return res;
}

/// @brief Update the RHS of the equation for finite difference fibers
void FiberContainerFiniteDifference::update_rhs(double dt, MatrixRef &v_on_fibers, MatrixRef &f_on_fibers) {
    size_t offset = 0;
    for (auto &fib : *this) {
        fib.update_RHS(dt, v_on_fibers.block(0, offset, 3, fib.n_nodes_),
                       f_on_fibers.block(0, offset, 3, fib.n_nodes_));
        offset += fib.n_nodes_;
    }
}

/// @brief Update the boundary conditions
void FiberContainerFiniteDifference::update_boundary_conditions(Periphery &shell,
                                                                const periphery_binding_t &periphery_binding) {
    for (auto &fib : *this) {
        fib.update_boundary_conditions(shell, periphery_binding);
    }
}

/// @brief Apply the boundary conditions
void FiberContainerFiniteDifference::apply_bcs(double dt, MatrixRef &v_on_fibers, MatrixRef &f_on_fibers) {
    // Call the underlying rectangular BC code
    apply_bc_rectangular(dt, v_on_fibers, f_on_fibers);
}

/// @brief Apply the rectangular BC
void FiberContainerFiniteDifference::apply_bc_rectangular(double dt, MatrixRef &v_on_fibers, MatrixRef &f_on_fibers) {
    size_t offset = 0;
    for (auto &fib : *this) {
        fib.apply_bc_rectangular(dt, v_on_fibers.block(0, offset, 3, fib.n_nodes_),
                                 f_on_fibers.block(0, offset, 3, fib.n_nodes_));
        fib.update_preconditioner();
        offset += fib.n_nodes_;
    }
}

/// Apply the fiber force
Eigen::MatrixXd FiberContainerFiniteDifference::apply_fiber_force(VectorRef &x_all) const {
    MatrixXd fw(3, x_all.size() / 4);

    size_t offset = 0;
    for (const auto &fib : *this) {
        const int np = fib.n_nodes_;
        Eigen::VectorXd force_fibers = fib.force_operator_ * x_all.segment(offset * 4, np * 4);
        fw.block(0, offset, 1, np) = force_fibers.segment(0 * np, np).transpose();
        fw.block(1, offset, 1, np) = force_fibers.segment(1 * np, np).transpose();
        fw.block(2, offset, 1, np) = force_fibers.segment(2 * np, np).transpose();

        offset += np;
    }

    return fw;
}

/// @brief Step fiber to new position according to current fiber solution
/// Updates: [fibers].x_
/// @param[in] fiber_sol [4 x n_nodes_tot] fiber solution vector
void FiberContainerFiniteDifference::step(VectorRef &fiber_sol) {
    size_t offset = 0;
    for (auto &fib : *this) {
        for (int i = 0; i < 3; ++i) {
            fib.x_.row(i) = fiber_sol.segment(offset, fib.n_nodes_);
            offset += fib.n_nodes_;
        }
        fib.tension_ = fiber_sol.segment(offset, fib.n_nodes_);
        offset += fib.n_nodes_;
    }
}

/// @brief Since the fiber and the body might not move _exactly_ the same, move the fiber to
/// lie exactly at the binding site again
/// Updates: [fibers].x_
/// @param[in] bodies BodyContainer object that contains fiber binding sites
void FiberContainerFiniteDifference::repin_to_bodies(BodyContainer &bodies) {
    for (auto &fib : *this) {
        if (fib.binding_site_.first >= 0) {
            Eigen::Vector3d delta =
                bodies.get_nucleation_site(fib.binding_site_.first, fib.binding_site_.second) - fib.x_.col(0);
            fib.x_.colwise() += delta;
        }
    }
}

/// @brief Get the RHS of the solution for FiberFiniteDifference.
VectorXd FiberContainerFiniteDifference::get_rhs() const {
    Eigen::VectorXd RHS(get_local_solution_size());
    int offset = 0;
    for (const auto &fib : *this) {
        RHS.segment(offset, fib.RHS_.size()) = fib.RHS_;
        offset += fib.RHS_.size();
    }

    return RHS;
}

/// @brief Apply the preconditioner to the fibers
Eigen::VectorXd FiberContainerFiniteDifference::apply_preconditioner(VectorRef &x_all) const {
    VectorXd y(x_all.size());
    size_t offset = 0;
    for (auto &fib : *this) {
        y.segment(offset, 4 * fib.n_nodes_) = fib.A_LU_.solve(x_all.segment(offset, 4 * fib.n_nodes_));
        offset += 4 * fib.n_nodes_;
    }
    return y;
}
