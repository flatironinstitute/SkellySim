#include <skelly_sim.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

#include <background_source.hpp>
#include <body.hpp>
#include <fiber_container_base.hpp>
#include <fiber_container_finite_difference.hpp>
#include <fiber_finite_difference.hpp>
#include <io_maps.hpp>
#include <params.hpp>
#include <parse_util.hpp>
#include <periphery.hpp>
#include <point_source.hpp>
#include <rng.hpp>
#include <solver_hydro.hpp>
#include <system.hpp>
#include <trajectory_reader.hpp>

#include <spdlog/cfg/env.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace System {
Params params_;                          ///< Simulation input parameters
std::unique_ptr<FiberContainerBase> fc_; ///< Fibers
BodyContainer bc_;                       ///< Bodies
PointSourceContainer psc_;               ///< Point Sources
BackgroundSource bs_;                    ///< Background flow

std::unique_ptr<Periphery> shell_; ///< Periphery
Eigen::VectorXd curr_solution_;    ///< Current MPI-rank local solution vector

std::unique_ptr<FiberContainerBase> fc_bak_; ///< Copy of fibers for timestep reversion
BodyContainer bc_bak_;                       ///< Copy of bodies for timestep reversion
int rank_;                                   ///< MPI rank
int size_;                                   ///< MPI size
toml::value param_table_;                    ///< Parsed input table

std::ofstream ofs_; ///< Trajectory output file stream. Opened at initialization

/// @brief external properties to the simulation
struct properties_t properties {
    // clang-format off
    .dt = 0.0,   ///< Current timestep
    .time = 0.0, ///< Current time
    // clang-format on
};

/// @brief Get current system-wide properties (time, timestep, etc.)
struct properties_t &get_properties() {
    return properties;
};

/// @brief Get current MPI-rank local solution vector
Eigen::VectorXd &get_curr_solution() { return curr_solution_; }

/// @brief Get number of physical nodes local to MPI rank for each object type [fibers, shell, bodies]
std::tuple<int, int, int> get_local_node_counts() {
    return std::make_tuple(fc_->get_local_node_count(), shell_->get_local_node_count(), bc_.get_local_node_count());
}

/// @brief Get GMRES solution size local to MPI rank for each object type [fibers, shell, bodies]
std::tuple<int, int, int> get_local_solution_sizes() {
    return std::make_tuple(fc_->get_local_solution_size(), shell_->get_local_solution_size(),
                           bc_.get_local_solution_size());
}

/// @brief Map 1D array data to a three-tuple of Vector Maps [fibers, shell, bodies]
std::tuple<VectorMap, VectorMap, VectorMap> get_solution_maps(double *x) {
    using Eigen::Map;
    using Eigen::VectorXd;
    auto [fib_sol_size, shell_sol_size, body_sol_size] = System::get_local_solution_sizes();
    return std::make_tuple(VectorMap(x, fib_sol_size), VectorMap(x + fib_sol_size, shell_sol_size),
                           VectorMap(x + fib_sol_size + shell_sol_size, body_sol_size));
}

/// @brief Map 1D array data to a three-tuple of const Vector Maps [fibers, shell, bodies]
std::tuple<CVectorMap, CVectorMap, CVectorMap> get_solution_maps(const double *x) {
    using Eigen::Map;
    using Eigen::VectorXd;
    auto [fib_sol_size, shell_sol_size, body_sol_size] = System::get_local_solution_sizes();
    return std::make_tuple(CVectorMap(x, fib_sol_size), CVectorMap(x + fib_sol_size, shell_sol_size),
                           CVectorMap(x + fib_sol_size + shell_sol_size, body_sol_size));
}

/// @brief Get size of local solution vector
std::size_t get_local_solution_size() {
    auto [fiber_sol_size, shell_sol_size, body_sol_size] = System::get_local_solution_sizes();
    return fiber_sol_size + shell_sol_size + body_sol_size;
}

/// @brief Flush current simulation state to ofstream
/// @param[in] ofs output stream to write to
void write(std::ofstream &ofs) {
    spdlog::trace("System::write");

    std::unique_ptr<FiberContainerBase> fc_global;
    BodyContainer bc_empty;
    BodyContainer &bc_global = (rank_ == 0) ? bc_ : bc_empty;
    Periphery shell_global;

    // Set up the type of FiberContainer we are going to serialize
    if (fc_->fiber_type_ == FiberContainerBase::FIBERTYPE::FiniteDifference) {
        fc_global = std::make_unique<FiberContainerFiniteDifference>();
    } else {
        throw std::runtime_error("Fiber discretization " + std::to_string(fc_->fiber_type_) +
                                 " used in write command, which does not exist.");
    }

    const output_map_t to_merge{properties.time, properties.dt, fc_, bc_global, *shell_, {RNG::dump_state()}};

    std::stringstream mergebuf;
    msgpack::pack(mergebuf, to_merge);

    std::string msg_local = mergebuf.str();
    int msgsize_local = msg_local.size();
    std::vector<int> msgsize(size_);
    MPI_Gather(&msgsize_local, 1, MPI_INT, msgsize.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    int msgsize_global = 0;
    std::vector<int> displs(size_ + 1);
    for (int i = 0; i < size_; ++i) {
        msgsize_global += msgsize[i];
        displs[i + 1] = displs[i] + msgsize[i];
    }

    std::vector<uint8_t> msg = (rank_ == 0) ? std::vector<uint8_t>(msgsize_global) : std::vector<uint8_t>();
    MPI_Gatherv(msg_local.data(), msgsize_local, MPI_CHAR, msg.data(), msgsize.data(), displs.data(), MPI_CHAR, 0,
                MPI_COMM_WORLD);

    shell_global.solution_vec_.resize(shell_->get_global_solution_size());

    if (rank_ == 0) {
        msgpack::object_handle oh;
        std::size_t offset = 0;

        output_map_t to_write{properties.time, properties.dt, fc_global, bc_global, shell_global};
        std::size_t shell_offset = 0;
        for (int i = 0; i < size_; ++i) {
            msgpack::unpack(oh, (char *)msg.data(), msg.size(), offset);
            msgpack::object obj = oh.get();
            input_map_t const &min_state = obj.as<input_map_t>();

            // FIXME: Get the new fiber implementation into its correct place
            if (min_state.fibers->fiber_type_ == FiberContainerBase::FIBERTYPE::FiniteDifference) {
                // Cast to correct type and do the fibers
                const FiberContainerFiniteDifference *fibers_fd =
                    static_cast<const FiberContainerFiniteDifference *>(min_state.fibers.get());
                FiberContainerFiniteDifference *fc_fd_global =
                    static_cast<FiberContainerFiniteDifference *>(fc_global.get());
                for (const auto &min_fib : fibers_fd->fibers_) {
                    fc_fd_global->fibers_.emplace_back(FiberFiniteDifference(min_fib, params_.eta));
                }
            } else {
                throw std::runtime_error("Fiber discretization " + std::to_string(min_state.fibers->fiber_type_) +
                                         " used in write command, which does not exist.");
            }

            // FIXME: WRANGLE IN THAT SHELL.SOLUTION now
            shell_global.solution_vec_.segment(shell_offset, min_state.shell.solution_vec_.size()) =
                min_state.shell.solution_vec_;
            shell_offset += min_state.shell.solution_vec_.size();

            to_write.rng_state.push_back(min_state.rng_state[0]);
        }

        msgpack::pack(ofs, to_write);
        ofs.flush();
    }

    spdlog::trace("System::write return");
}

/// @brief Dump current state to single file
///
/// @param[in] config_file path of file to output
void write_config(const std::string &config_file) {
    auto trajectory_open_mode = std::ofstream::binary | std::ofstream::out;
    auto ofs = std::ofstream(config_file, trajectory_open_mode);
    write(ofs);
}

/// @brief Write a header that contains the trajectory version information
///
/// @param[in] ofs output stream to write to
void write_header(std::ofstream &ofs) {
    spdlog::trace("System::write_header");

    // We just need the header file information
    auto now = std::chrono::system_clock::now();
    std::time_t current_time = std::chrono::system_clock::to_time_t(now);
    auto simdate = std::ctime(&current_time);
    // FIXME XXX This only works on linux machines, but I don't think anybody is going to be using SkellySim on windows
    // anytime soon...
    char c_hostname[1024];
    gethostname(c_hostname, sizeof(c_hostname));
    std::string hostname = c_hostname;
    const header_map_t to_write{SKELLYSIM_TRAJECTORY_VERSION,
                                size_,
                                (int)fc_->fiber_type_,
                                SKELLYSIM_VERSION,
                                SKELLYSIM_COMMIT,
                                simdate,
                                hostname};

    // Make sure we are just on rank 0 for the output
    if (rank_ == 0) {
        msgpack::pack(ofs, to_write);
        ofs.flush();
    }

    spdlog::trace("System::write_header return");
}

/// @brief Set system state to last state found in trajectory files
///
/// @param[in] input_file input file name of trajectory file for this rank
void resume_from_trajectory(std::string input_file) {
    TrajectoryReader trajectory(input_file, true);
    while (trajectory.read_next_frame()) {
    }
    trajectory.unpack_current_frame();
}

/// @brief Map Eigen Matrix node data to a three-tuple of Matrix Block references (use like view)
///
/// @param[in] x [3 x n_nodes_local] Matrix where you want the views
/// @return Three-tuple of node data [3 x n_fiber_nodes, 3 x n_bodiy_nodes, 3 x n_periphery_nodes]
template <typename Derived>
std::tuple<Eigen::Block<Derived>, Eigen::Block<Derived>, Eigen::Block<Derived>>
get_node_maps(Eigen::MatrixBase<Derived> &x) {
    auto [fib_nodes, shell_nodes, body_nodes] = get_local_node_counts();
    return std::make_tuple(Eigen::Block<Derived>(x.derived(), 0, 0, 3, fib_nodes),
                           Eigen::Block<Derived>(x.derived(), 0, fib_nodes, 3, shell_nodes),
                           Eigen::Block<Derived>(x.derived(), 0, fib_nodes + shell_nodes, 3, body_nodes));
}

/// @brief Apply and return preconditioner results from fibers/body/shell
///
/// \f[ P^{-1} * x = y \f]
/// @param [in] x [local_solution_size] Vector to apply preconditioner on
/// @return [local_solution_size] Preconditioned input vector
Eigen::VectorXd apply_preconditioner(CVectorRef &x) {
    const auto [fib_sol_size, shell_sol_size, body_sol_size] = get_local_solution_sizes();
    const int sol_size = fib_sol_size + shell_sol_size + body_sol_size;
    assert(sol_size == x.size());
    Eigen::VectorXd res(sol_size);

    auto [x_fibers, x_shell, x_bodies] = get_solution_maps(x.data());
    auto [res_fibers, res_shell, res_bodies] = get_solution_maps(res.data());

    res_fibers = fc_->apply_preconditioner(x_fibers);
    res_shell = shell_->apply_preconditioner(x_shell);
    res_bodies = bc_.apply_preconditioner(x_bodies);

    return res;
}

/// @brief Apply and return entire operator on system state vector for fibers/body/shell
///
/// \f[ A * x = y \f]
/// @param [in] x [local_solution_size] Vector to apply matvec on
/// @return [local_solution_size] Vector y, the result of the operator applied to x.
Eigen::VectorXd apply_matvec(CVectorRef &x) {
    spdlog::trace("System::apply_matvec");

    using Eigen::Block;
    using Eigen::MatrixXd;
    const FiberContainerBase &fc = *fc_;
    const Periphery &shell = *shell_;
    const BodyContainer &bc = bc_;
    const double eta = params_.eta;

    const auto [fib_node_count, shell_node_count, body_node_count] = get_local_node_counts();
    const int total_node_count = fib_node_count + shell_node_count + body_node_count;

    const auto [fib_sol_size, shell_sol_size, body_sol_size] = get_local_solution_sizes();
    const int sol_size = fib_sol_size + shell_sol_size + body_sol_size;
    assert(sol_size == x.size());
    Eigen::VectorXd res(sol_size);

    MatrixXd r_all(3, total_node_count), v_all(3, total_node_count);
    auto [r_fibers, r_shell, r_bodies] = get_node_maps(r_all);
    auto [v_fibers, v_shell, v_bodies] = get_node_maps(v_all);
    r_fibers = fc.get_local_node_positions();
    r_shell = shell.get_local_node_positions();
    r_bodies = bc.get_local_node_positions(bc.bodies);

    auto [x_fibers, x_shell, x_bodies] = get_solution_maps(x.data());
    auto [res_fibers, res_shell, res_bodies] = get_solution_maps(res.data());

    // calculate fiber-fiber velocity
    MatrixXd fw = fc.apply_fiber_force(x_fibers);
    MatrixXd v_fib2all = fc.flow(r_all, fw, eta);

    MatrixXd r_fibbody(3, r_fibers.cols() + r_bodies.cols());
    r_fibbody.block(0, 0, 3, r_fibers.cols()) = r_fibers;
    r_fibbody.block(0, r_fibers.cols(), 3, r_bodies.cols()) = r_bodies;
    MatrixXd v_shell2fibbody = shell.flow(r_fibbody, x_shell, eta);

    Eigen::VectorXd x_bodies_global(bc.get_global_solution_size());
    if (rank_ == 0)
        x_bodies_global = x_bodies;
    MPI_Bcast(x_bodies_global.data(), x_bodies_global.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MatrixXd fiber_link_conditions, body_link_conditions;
    std::tie(fiber_link_conditions, body_link_conditions) = fc.calculate_link_conditions(x_fibers, x_bodies_global, bc);

    v_all = v_fib2all;
    v_fibers += v_shell2fibbody.block(0, 0, 3, r_fibers.cols());
    v_bodies += v_shell2fibbody.block(0, r_fibers.cols(), 3, r_bodies.cols());
    v_all += bc.flow(r_all, x_bodies, body_link_conditions, eta);

    res_fibers = fc.matvec(x_fibers, v_fibers, fiber_link_conditions);
    res_shell = shell.matvec(x_shell, v_shell);
    res_bodies = bc.matvec(v_bodies, x_bodies);

    spdlog::trace("System::apply_matvec return");
    return res;
}

/// @brief Evaluate the velocity at a list of target points
///
/// @param[in] r_trg [3 x n_trg] matrix of points to evaluate velocities
/// @return [3 x n_trg] matrix of velocities at r_trg
Eigen::MatrixXd velocity_at_targets(CMatrixRef &r_trg) {
    if (!r_trg.size())
        return Eigen::MatrixXd(3, 0);
    Eigen::MatrixXd u_trg(r_trg.rows(), r_trg.cols());

    const double eta = params_.eta;
    const auto [sol_fibers, sol_shell, sol_bodies] = get_solution_maps(curr_solution_.data());
    const auto &fp = params_.fiber_periphery_interaction;

    Eigen::MatrixXd f_on_fibers = fc_->apply_fiber_force(sol_fibers);
    if (params_.periphery_interaction_flag)
        f_on_fibers += fc_->periphery_force(*shell_, fp);

    // FIXME: This is likely wrong, but more right than before
    Eigen::VectorXd sol_bodies_global(bc_.get_global_solution_size());
    if (rank_ == 0)
        sol_bodies_global = sol_bodies;
    MPI_Bcast(sol_bodies_global.data(), sol_bodies_global.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // This routine zeros out the external force. is that correct?
    auto [fiber_link_conditions, body_link_conditions] =
        fc_->calculate_link_conditions(sol_fibers, sol_bodies_global, bc_);

    Eigen::MatrixXd body_forces_torques = bc_.calculate_external_forces_torques(properties.time);

    // clang-format off
    u_trg = fc_->flow(r_trg, f_on_fibers, eta, false) + \
        bc_.flow(r_trg, sol_bodies, body_link_conditions, eta) + \
        shell_->flow(r_trg, sol_shell, eta) + \
        psc_.flow(r_trg, eta, properties.time) + \
        bs_.flow(r_trg, eta);
    // clang-format on

    // FIXME: move this to body logic with overloading
    // Replace points inside a body to have velocity v_body + w_body x r
    for (int i = 0; i < r_trg.cols(); ++i) {
        for (auto &body : bc_.spherical_bodies) {
            Eigen::Vector3d dx = r_trg.col(i) - body->position_;
            if (dx.norm() < body->radius_)
                u_trg.col(i) = body->velocity_ + body->angular_velocity_.cross(dx);
        }
        // FIXME: There would be something here if we had flow for deformable bodies
        for (auto &body : bc_.ellipsoidal_bodies) {
            Eigen::Vector3d dx = r_trg.col(i) - body->position_;
            // Actually have to calculate the real dx inside the ellipsoid
            if (dx[0] * dx[0] / body->radius_[0] / body->radius_[0] +
                    dx[1] * dx[1] / body->radius_[1] / body->radius_[1] +
                    dx[2] * dx[2] / body->radius_[2] / body->radius_[2] <
                1.0) {
                u_trg.col(i) = body->velocity_ + body->angular_velocity_.cross(dx);
            }
        }
    }

    return u_trg;
}

/// @brief Change the pair interaction evaluator method
///
/// @param[in] evaluator (FMM, GPU, CPU)
void set_evaluator(const std::string &evaluator) {
    fc_->set_evaluator(evaluator);
    bc_.set_evaluator(evaluator);
    shell_->set_evaluator(evaluator);
}

/// @brief Calculate all initial velocities/forces/RHS/BCs
///
/// @note Modifies anything that evolves in time.
void prep_state_for_solver() {
    spdlog::trace("System::prep_state_for_solver");
    using Eigen::MatrixXd;

    // Since DI can change size of fiber containers, must call first.
    System::dynamic_instability();
    fc_->update_cache_variables(properties.dt, params_.eta);

    const auto [fib_node_count, shell_node_count, body_node_count] = get_local_node_counts();

    MatrixXd r_all(3, fib_node_count + shell_node_count + body_node_count);
    {
        auto [r_fibers, r_shell, r_bodies] = get_node_maps(r_all);
        r_fibers = fc_->get_local_node_positions();
        r_shell = shell_->get_local_node_positions();
        r_bodies = bc_.get_local_node_positions(bc_.bodies);
    }

    // Implicit motor forces
    MatrixXd motor_force_fibers = params_.implicit_motor_activation_delay > properties.time
                                      ? MatrixXd::Zero(3, fib_node_count)
                                      : fc_->generate_constant_force();

    // Fiber-periphery forces (if periphery exists)
    MatrixXd external_force_fibers = fc_->periphery_force(*shell_, params_.fiber_periphery_interaction);

    // Don't include motor forces for initial calculation (explicitly handled elsewhere)
    MatrixXd v_all = fc_->flow(r_all, external_force_fibers, params_.eta);

    bc_.update_cache_variables(params_.eta);

    // Check for an add external body forces
    // FIXME: Calculates local forces/torques to each rank. all-reduced during flow, so hackishly predivide here
    MatrixXd body_forces_torques = bc_.calculate_external_forces_torques(properties.time) / size_;

    if (body_forces_torques.any()) {
        const int total_node_count = fib_node_count + shell_node_count + body_node_count;
        MatrixXd r_all(3, total_node_count);
        auto [r_fibers, r_shell, r_bodies] = get_node_maps(r_all);
        r_fibers = fc_->get_local_node_positions();
        r_shell = shell_->get_local_node_positions();
        r_bodies = bc_.get_local_node_positions(bc_.bodies);

        v_all +=
            bc_.flow(r_all, Eigen::VectorXd::Zero(bc_.get_local_solution_size()), body_forces_torques, params_.eta);
    }

    v_all += psc_.flow(r_all, params_.eta, properties.time);
    v_all += bs_.flow(r_all, params_.eta);

    bc_.update_RHS(v_all.block(0, fib_node_count + shell_node_count, 3, body_node_count));

    MatrixXd total_force_fibers = motor_force_fibers + external_force_fibers;
    fc_->update_rhs(properties.dt, v_all.block(0, 0, 3, fib_node_count), total_force_fibers);
    fc_->update_boundary_conditions(*shell_, params_.periphery_binding);
    fc_->apply_bcs(properties.dt, v_all.block(0, 0, 3, fib_node_count), external_force_fibers);

    shell_->update_RHS(v_all.block(0, fib_node_count, 3, shell_node_count));

    spdlog::trace("System::prep_state_for_solver return");
}

/// @brief Calculate solution vector given current configuration
///
/// @note Modifies too much stuff to note reasonably. RHS info, cache info, and current solution, most notably
/// @return If the solver converged to the requested tolerance with no issue.
bool solve() {
    prep_state_for_solver();

    Solver<P_inv_hydro, A_fiber_hydro> solver_; /// < Wrapper class for solving system
    solver_.set_RHS();
    // FIXME: Rename this, as it just packs up things for the solver, something like grab_RHS_from_system

    bool converged = solver_.solve();
    curr_solution_ = solver_.get_solution();

    spdlog::info("Residual: {}", solver_.get_residual());
    return converged;
}

/// @brief Generate next trial system state for the current System::properties::dt
///
/// @note Modifies anything that evolves in time.
/// @return If the solver converged to the requested tolerance with no issue.
bool step() {
    bool converged = solve();
    auto [fiber_sol, shell_sol, body_sol] = get_solution_maps(curr_solution_.data());

    fc_->step(fiber_sol);
    bc_.step(body_sol, properties.dt);
    fc_->repin_to_bodies(bc_);
    shell_->step(shell_sol);

    return converged;
}

/// @brief store copies of Fiber and Body containers in case time step is rejected
void backup() {
    // We have to decode what kind of fiber container we have to restore the backup
    if (fc_->fiber_type_ == FiberContainerBase::FIBERTYPE::FiniteDifference) {
        *fc_bak_ = *fc_;
    } else {
        throw std::runtime_error("Fiber type " + std::to_string(fc_->fiber_type_) + " not implemented for backup");
    }
    bc_bak_ = bc_;
}

/// @brief restore copies of Fiber and Body containers to the state when last backed up
void restore() {
    if (fc_bak_->fiber_type_ == FiberContainerBase::FIBERTYPE::FiniteDifference) {
        *fc_ = *fc_bak_;
    } else {
        throw std::runtime_error("Fiber type " + std::to_string(fc_->fiber_type_) + " not implemented for restore");
    }
    bc_ = bc_bak_;
}

/// @brief Run the simulation!
void run() {
    Params &params = params_;

    while (properties.time < params.t_final) {
        // Store system state so we can revert if the timestep fails
        System::backup();
        // Run the system timestep and store convergence
        bool converged = System::step();
        // Maximum error in the fiber derivative
        double fiber_error = fc_->fiber_error_local();
        MPI_Allreduce(MPI_IN_PLACE, &fiber_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        double dt_new = properties.dt;
        bool accept = false;
        if (params.adaptive_timestep_flag) {
            // Now all the acceptance/adaptive timestepping logic
            if (converged && fiber_error <= params.fiber_error_tol) {
                accept = true;
                const double tol_window = 0.9 * params.fiber_error_tol;
                if (fiber_error <= tol_window)
                    dt_new = std::min(params.dt_max, properties.dt * params.beta_up);
            } else {
                dt_new = properties.dt * params.beta_down;
                accept = false;
            }

            if (converged && System::check_collision()) {
                spdlog::info("Collision detected, rejecting solution and taking a smaller timestep");
                dt_new = properties.dt * 0.5;
                accept = false;
            }

            if (dt_new < params.dt_min) {
                spdlog::info("System time, dt, fiber_error: {}, {}, {}", properties.time, dt_new, fiber_error);
                spdlog::critical("Timestep smaller than minimum allowed");
                throw std::runtime_error("Timestep smaller than dt_min");
            }

            properties.dt = dt_new;
        }
        if (!params.adaptive_timestep_flag || accept) {
            spdlog::info("Accepting timestep and advancing time");
            properties.time += properties.dt;
            double &dt_write = params_.dt_write;
            if ((int)(properties.time / dt_write) > (int)((properties.time - properties.dt) / dt_write))
                write(ofs_);
        } else {
            spdlog::info("Rejecting timestep");
            System::restore();
        }

        spdlog::info("System time, dt, fiber_error: {}, {}, {}", properties.time, dt_new, fiber_error);
    }

    write_config("skelly_sim.final_config");
}

/// @brief Check for any collisions between objects
///
/// @return true if any collision detected, false otherwise
bool check_collision() {
    const double threshold = 0.0;
    using Eigen::VectorXd;

    char collided = false;
    for (const auto &body : bc_.bodies)
        if (!collided && body->check_collision(*shell_, threshold))
            collided = true;

    collided = collided || fc_->check_collision(*shell_, threshold);

    for (auto &body1 : bc_.bodies)
        for (auto &body2 : bc_.bodies)
            if (!collided && body1 != body2 && body1->check_collision(*body2, threshold))
                collided = true;

    MPI_Allreduce(MPI_IN_PLACE, &collided, 1, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);

    return collided;
}

/// @brief Return copy of fiber container's RHS
Eigen::VectorXd get_fiber_RHS() { return fc_->get_rhs(); }
/// @brief Return copy of body container's RHS
Eigen::VectorXd get_body_RHS() { return bc_.get_RHS(); }
/// @brief Return copy of shell's RHS
Eigen::VectorXd get_shell_RHS() { return shell_->get_RHS(); }

/// @brief get pointer to params struct
Params *get_params() { return &params_; }
/// @brief get pointer to body container
BodyContainer *get_body_container() { return &bc_; }
/// @brief get pointer to fiber container
FiberContainerBase *get_fiber_container() { return fc_.get(); }
/// @brief get pointer to shell
Periphery *get_shell() { return shell_.get(); }
/// @brief get pointer to param table struct
toml::value *get_param_table() { return &param_table_; }
/// @brief get pointer to point source container
PointSourceContainer *get_point_source_container() { return &psc_; }

/// @brief Raise relevant exception if known conflict in parameter setup
void sanity_check() {
    if ((params_.pair_evaluator == "CPU" || params_.pair_evaluator == "GPU") && size_ > 1) {
        throw std::runtime_error("More than one MPI rank, but \"" + params_.pair_evaluator +
                                 "\" supplied as pair evaluator. Only \"FMM\" is a "
                                 "valid MPI evaluator currently. ");
    }

    if (shell_->is_active() && bs_.is_active())
        throw std::runtime_error("Background sources are currently incompatible with peripheries.");
}

/// @brief Initialize entire system. Needs to be called once at the beginning of the program execution
/// @param[in] input_file String of toml config file specifying system parameters and initial conditions
/// @param[in] resume_flag true if simulation is resuming from prior execution state, false otherwise.
void init(const std::string &input_file, bool resume_flag, bool listen_flag) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    spdlog::logger sink = rank_ == 0
                              ? spdlog::logger("SkellySim", std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>())
                              : spdlog::logger("SkellySim", std::make_shared<spdlog::sinks::null_sink_st>());
    spdlog::set_default_logger(std::make_shared<spdlog::logger>(sink));
    spdlog::stderr_color_mt("STKFMM");
    spdlog::stderr_color_mt("Belos");
    spdlog::stderr_color_mt("SkellySim global");
    spdlog::cfg::load_env_levels();

    // Now that we have a logger, we can trace input and stuff
    spdlog::trace("System::init");

    param_table_ = toml::parse(input_file);
    params_ = Params(param_table_.at("params"));
    params_.print();
    RNG::init(params_.seed);

    properties.dt = params_.dt_initial;

    // New fiber containers
    spdlog::trace("Creating FiberContainerBase");
    if (param_table_.contains("fibers")) {
        if (params_.fiber_type == "FiniteDifference") {
            fc_ = std::make_unique<FiberContainerFiniteDifference>(param_table_.at("fibers").as_array(), params_);
            // Create the empty version of this too for the backup
            fc_bak_ = std::make_unique<FiberContainerFiniteDifference>();
        } else if (params_.fiber_type == "None") {
            throw std::runtime_error("Fibers found but no fiber discretization specified.");
        } else {
            throw std::runtime_error("Fibers found but incorrect fiber discretization " + params_.fiber_type +
                                     " specified.");
        }
    } else {
        spdlog::trace("  Creating an empty FiberContainerFiniteDifference (no fibers)");
        // Default will be an empty finite difference fiber setup, as the base class doesn't know anything, and may
        // error if compiled in and run
        fc_ = std::make_unique<FiberContainerFiniteDifference>();
        // Backup version
        fc_bak_ = std::make_unique<FiberContainerFiniteDifference>();
    }

    if (param_table_.contains("periphery")) {
        const toml::value &periphery_table = param_table_.at("periphery");
        if (toml::find_or(periphery_table, "shape", "") == std::string("sphere"))
            shell_ = std::make_unique<SphericalPeriphery>(periphery_table, params_);
        else if (toml::find_or(periphery_table, "shape", "") == std::string("ellipsoid"))
            shell_ = std::make_unique<EllipsoidalPeriphery>(periphery_table, params_);
        else // Assume generic periphery for all other shapes
            shell_ = std::make_unique<GenericPeriphery>(periphery_table, params_);
    } else {
        shell_ = std::make_unique<Periphery>();
    }

    if (param_table_.contains("bodies"))
        bc_ = BodyContainer(param_table_.at("bodies").as_array(), params_);

    if (param_table_.contains("point_sources"))
        psc_ = PointSourceContainer(param_table_.at("point_sources").as_array());

    if (param_table_.contains("background"))
        bs_ = BackgroundSource(param_table_.at("background"));

    sanity_check();

    curr_solution_.resize(get_local_solution_size());
    std::string filename = "skelly_sim.out";
    auto trajectory_open_mode = std::ofstream::binary | (listen_flag ? std::ofstream::in : std::ofstream::out);
    if (resume_flag) {
        resume_from_trajectory(filename);
        trajectory_open_mode = trajectory_open_mode | std::ofstream::app;
    }
    if (rank_ == 0) {
        ofs_ = std::ofstream(filename, trajectory_open_mode);
        // If we are not a resume, then dump the header information into the trajectory file
        if (!resume_flag && !listen_flag) {
            write_header(ofs_);
        }
    }

    // Do not write out an initial config if listening!
    if (!listen_flag) {
        write_config("skelly_sim.initial_config");
    }

    spdlog::trace("System::init return");
}
} // namespace System
