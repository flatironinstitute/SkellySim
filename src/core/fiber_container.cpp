#include <skelly_sim.hpp>

#include <algorithm>

#include <fiber.hpp>
#include <kernels.hpp>
#include <periphery.hpp>
#include <system.hpp>
#include <utils.hpp>

using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

/// @brief Check if each fiber configuration is within some bounds for attachment with periphery
///
/// Updates: Fiber::bc_minus, Fiber::bc_plus
/// @param[in] Periphery object
void FiberContainer::update_boundary_conditions(Periphery &shell) {
    for (auto &fib : *this)
        fib.update_boundary_conditions(shell);
}

/// @brief Get total number of fibers across all ranks
/// @return total number of fibers across all ranks
int FiberContainer::get_global_count() const {
    const int local_fib_count = get_local_count();
    int global_fib_count;

    MPI_Allreduce(&local_fib_count, &global_fib_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return global_fib_count;
}

/// @brief Get total number of fiber nodes across all ranks
/// @return total number of fiber nodes across all ranks
int FiberContainer::get_global_total_fib_nodes() const {
    const int local_fib_nodes = get_local_node_count();
    int global_fib_nodes;

    MPI_Allreduce(&local_fib_nodes, &global_fib_nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return global_fib_nodes;
}

/// @brief Update derivatives on all fibers
/// See: Fiber::update_derivatives
void FiberContainer::update_derivatives() {
    for (auto &fib : *this)
        fib.update_derivatives();
}

void FiberContainer::update_stokeslets(double eta) {
    // FIXME: Remove default arguments for stokeslets
    for (auto &fib : *this)
        fib.update_stokeslet(eta);
}

void FiberContainer::update_linear_operators(double dt, double eta) {
    for (auto &fib : *this)
        fib.update_linear_operator(dt, eta);
}

VectorXd FiberContainer::apply_preconditioner(VectorRef &x_all) const {
    VectorXd y(x_all.size());
    size_t offset = 0;
    for (auto &fib : *this) {
        y.segment(offset, 4 * fib.n_nodes_) = fib.A_LU_.solve(x_all.segment(offset, 4 * fib.n_nodes_));
        offset += 4 * fib.n_nodes_;
    }
    return y;
}

VectorXd FiberContainer::matvec(VectorRef &x_all, MatrixRef &v_fib) const {
    VectorXd res = VectorXd::Zero(get_local_solution_size());

    size_t offset = 0;
    int i_fib = 0;
    for (const auto &fib : *this) {
        const int np = fib.n_nodes_;
        res.segment(offset, 4 * np) = fib.matvec(x_all.segment(offset, 4 * np), v_fib.block(0, i_fib, 3, np));

        i_fib++;
        offset += 4 * np;
    }

    return res;
}

VectorXd FiberContainer::get_RHS() const {
    Eigen::VectorXd RHS(get_local_solution_size());
    int offset = 0;
    for (const auto &fib : *this) {
        RHS.segment(offset, fib.RHS_.size()) = fib.RHS_;
        offset += fib.RHS_.size();
    }

    return RHS;
}

MatrixXd FiberContainer::flow(const MatrixRef &r_trg, const MatrixRef &fib_forces, double eta,
                              bool subtract_self) const {
    spdlog::debug("Starting fiber flow");
    const size_t n_src = fib_forces.cols();
    const size_t n_trg = r_trg.cols();
    if (!get_global_count())
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

    spdlog::debug("Finished fiber flow");
    return vel;
}

MatrixXd FiberContainer::generate_constant_force() const {
    const int n_fib_pts = get_local_node_count();
    MatrixXd f(3, n_fib_pts);
    size_t offset = 0;
    for (const auto &fib : *this) {
        f.block(0, offset, 3, fib.n_nodes_) = fib.force_scale_ * fib.xs_;
        offset += fib.n_nodes_;
    }
    return f;
}

MatrixXd FiberContainer::apply_fiber_force(VectorRef &x_all) const {
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

void FiberContainer::update_local_node_positions() {
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

void FiberContainer::update_cache_variables(double dt, double eta) {
    for (auto &fib : *this) {
        fib.update_constants(eta);
        fib.update_derivatives();
        fib.update_stokeslet(eta);
        fib.update_linear_operator(dt, eta);
        fib.update_force_operator();
    }

    update_local_node_positions();
}

void FiberContainer::update_RHS(double dt, MatrixRef &v_on_fibers, MatrixRef &f_on_fibers) {
    size_t offset = 0;
    for (auto &fib : *this) {
        fib.update_RHS(dt, v_on_fibers.block(0, offset, 3, fib.n_nodes_),
                       f_on_fibers.block(0, offset, 3, fib.n_nodes_));
        offset += fib.n_nodes_;
    }
}

void FiberContainer::apply_bc_rectangular(double dt, MatrixRef &v_on_fibers, MatrixRef &f_on_fibers) {
    size_t offset = 0;
    for (auto &fib : *this) {
        fib.apply_bc_rectangular(dt, v_on_fibers.block(0, offset, 3, fib.n_nodes_),
                                 f_on_fibers.block(0, offset, 3, fib.n_nodes_));
        fib.update_preconditioner();
        offset += fib.n_nodes_;
    }
}

/// @brief Step fiber to new position according to current fiber solution
/// Updates: [fibers].x_
/// @param[in] fiber_sol [4 x n_nodes_tot] fiber solution vector
void FiberContainer::step(VectorRef &fiber_sol) {
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


void FiberContainer::set_evaluator(const std::string &evaluator) {
    auto &params = *System::get_params();

    if (evaluator == "FMM") {
        utils::LoggerRedirect redirect(std::cout);
        const int mult_order = params.stkfmm.fiber_stokeslet_multipole_order;
        const int max_pts = params.stkfmm.fiber_stokeslet_max_points;
        stokeslet_kernel_ = kernels::FMM<stkfmm::Stk3DFMM>(mult_order, max_pts, stkfmm::PAXIS::NONE,
                                                           stkfmm::KERNEL::Stokes, kernels::stokes_vel_fmm);
        redirect.flush(spdlog::level::debug, "STKFMM");
    } else if (evaluator == "CPU")
        stokeslet_kernel_ = kernels::stokeslet_direct_cpu;
    else if (evaluator == "GPU")
        stokeslet_kernel_ = kernels::stokeslet_direct_gpu;
}

FiberContainer::FiberContainer(Params &params) {
    spdlog::info("Initializing FiberContainer");
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);

    set_evaluator(params.pair_evaluator);
}

FiberContainer::FiberContainer(toml::array &fiber_tables, Params &params) {
    *this = FiberContainer(params);

    const int n_fibs_tot = fiber_tables.size();
    const int n_fibs_extra = n_fibs_tot % world_size_;
    spdlog::info("Reading in {} fibers.", n_fibs_tot);

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
            fibers.emplace_back(fiber_table, params.eta);

            auto &fib = fibers.back();
            spdlog::get("SkellySim global")
                ->debug("Fiber {}: {} {} {}", i_fib, fib.n_nodes_, fib.bending_rigidity_, fib.length_);
        }
    }

    update_local_node_positions();
}
