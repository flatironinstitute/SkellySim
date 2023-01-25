#include <skelly_sim.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <background_source.hpp>
#include <fiber.hpp>
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
Params params_;            ///< Simulation input parameters
FiberContainer fc_;        ///< Fibers
PointSourceContainer psc_; ///< Point Sources
BackgroundSource bs_;      ///< Background flow

std::unique_ptr<Periphery> shell_; ///< Periphery
Eigen::VectorXd curr_solution_;    ///< Current MPI-rank local solution vector

FiberContainer fc_bak_;   ///< Copy of fibers for timestep reversion
int rank_;                ///< MPI rank
int size_;                ///< MPI size
toml::value param_table_; ///< Parsed input table

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
std::tuple<int, int> get_local_node_counts() {
    return std::make_tuple(fc_.get_local_node_count(), shell_->get_local_node_count());
}

/// @brief Get GMRES solution size local to MPI rank for each object type [fibers, shell, bodies]
std::tuple<int, int> get_local_solution_sizes() {
    return std::make_tuple(fc_.get_local_solution_size(), shell_->get_local_solution_size());
}

/// @brief Map 1D array data to a three-tuple of Vector Maps [fibers, shell, bodies]
std::tuple<VectorMap, VectorMap> get_solution_maps(double *x) {
    using Eigen::Map;
    using Eigen::VectorXd;
    auto [fib_sol_size, shell_sol_size] = System::get_local_solution_sizes();
    return std::make_tuple(VectorMap(x, fib_sol_size), VectorMap(x + fib_sol_size, shell_sol_size));
}

/// @brief Map 1D array data to a three-tuple of const Vector Maps [fibers, shell, bodies]
std::tuple<CVectorMap, CVectorMap> get_solution_maps(const double *x) {
    using Eigen::Map;
    using Eigen::VectorXd;
    auto [fib_sol_size, shell_sol_size] = System::get_local_solution_sizes();
    return std::make_tuple(CVectorMap(x, fib_sol_size), CVectorMap(x + fib_sol_size, shell_sol_size));
}

/// @brief Get size of local solution vector
std::size_t get_local_solution_size() {
    auto [fiber_sol_size, shell_sol_size] = System::get_local_solution_sizes();
    return fiber_sol_size + shell_sol_size;
}

/// @brief Flush current simulation state to ofstream
/// @param[in] ofs output stream to write to
void write(std::ofstream &ofs) {
    FiberContainer fc_global;
    Periphery shell_global;

    const output_map_t to_merge{properties.time, properties.dt, fc_, *shell_, {RNG::dump_state()}};

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

        output_map_t to_write{properties.time, properties.dt, fc_global, shell_global};
        std::size_t shell_offset = 0;
        for (int i = 0; i < size_; ++i) {
            msgpack::unpack(oh, (char *)msg.data(), msg.size(), offset);
            msgpack::object obj = oh.get();
            input_map_t const &min_state = obj.as<input_map_t>();
            for (const auto &min_fib : min_state.fibers.fibers)
                fc_global.fibers.emplace_back(Fiber(min_fib, params_.eta));

            // FIXME: WRANGLE IN THAT SHELL.SOLUTION now
            shell_global.solution_vec_.segment(shell_offset, min_state.shell.solution_vec_.size()) =
                min_state.shell.solution_vec_;
            shell_offset += min_state.shell.solution_vec_.size();

            to_write.rng_state.push_back(min_state.rng_state[0]);
        }

        msgpack::pack(ofs, to_write);
        ofs.flush();
    }
}

/// @brief Dump current state to single file
///
/// @param[in] config_file path of file to output
void write_config(const std::string &config_file) {
    auto trajectory_open_mode = std::ofstream::binary | std::ofstream::out;
    auto ofs = std::ofstream(config_file, trajectory_open_mode);
    write(ofs);
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
std::tuple<Eigen::Block<Derived>, Eigen::Block<Derived>>
get_node_maps(Eigen::MatrixBase<Derived> &x) {
    auto [fib_nodes, shell_nodes] = get_local_node_counts();
    return std::make_tuple(Eigen::Block<Derived>(x.derived(), 0, 0, 3, fib_nodes),
                           Eigen::Block<Derived>(x.derived(), 0, fib_nodes, 3, shell_nodes));
}

/// @brief Apply and return preconditioner results from fibers/shell
///
/// \f[ P^{-1} * x = y \f]
/// @param [in] x [local_solution_size] Vector to apply preconditioner on
/// @return [local_solution_size] Preconditioned input vector
Eigen::VectorXd apply_preconditioner(VectorRef &x) {
    const auto [fib_sol_size, shell_sol_size] = get_local_solution_sizes();
    const int sol_size = fib_sol_size + shell_sol_size;
    assert(sol_size == x.size());
    Eigen::VectorXd res(sol_size);

    auto [x_fibers, x_shell] = get_solution_maps(x.data());
    auto [res_fibers, res_shell] = get_solution_maps(res.data());

    res_fibers = fc_.apply_preconditioner(x_fibers);
    res_shell = shell_->apply_preconditioner(x_shell);

    return res;
}

/// @brief Apply and return entire operator on system state vector for fibers/shell
///
/// \f[ A * x = y \f]
/// @param [in] x [local_solution_size] Vector to apply matvec on
/// @return [local_solution_size] Vector y, the result of the operator applied to x.
Eigen::VectorXd apply_matvec(VectorRef &x) {
    using Eigen::Block;
    using Eigen::MatrixXd;
    const FiberContainer &fc = fc_;
    const Periphery &shell = *shell_;
    const double eta = params_.eta;

    const auto [fib_node_count, shell_node_count] = get_local_node_counts();
    const int total_node_count = fib_node_count + shell_node_count;

    const auto [fib_sol_size, shell_sol_size] = get_local_solution_sizes();
    const int sol_size = fib_sol_size + shell_sol_size;
    assert(sol_size == x.size());
    Eigen::VectorXd res(sol_size);

    MatrixXd r_all(3, total_node_count), v_all(3, total_node_count);
    auto [r_fibers, r_shell] = get_node_maps(r_all);
    auto [v_fibers, v_shell] = get_node_maps(v_all);
    r_fibers = fc.get_local_node_positions();
    r_shell = shell.get_local_node_positions();

    auto [x_fibers, x_shell] = get_solution_maps(x.data());
    auto [res_fibers, res_shell] = get_solution_maps(res.data());

    // calculate fiber-fiber velocity
    MatrixXd fw = fc.apply_fiber_force(x_fibers);
    MatrixXd v_fib2all = fc.flow(r_all, fw, eta);
    MatrixXd v_shell2fibers = shell.flow(r_fibers, x_shell, eta);

    v_all = v_fib2all;
    v_fibers += v_shell2fibers;

    res_fibers = fc.matvec(x_fibers, v_fibers);
    res_shell = shell.matvec(x_shell, v_shell);

    return res;
}

/// @brief Evaluate the velocity at a list of target points
///
/// @param[in] r_trg [3 x n_trg] matrix of points to evaluate velocities
/// @return [3 x n_trg] matrix of velocities at r_trg
Eigen::MatrixXd velocity_at_targets(MatrixRef &r_trg) {
    if (!r_trg.size())
        return Eigen::MatrixXd(3, 0);
    Eigen::MatrixXd u_trg(r_trg.rows(), r_trg.cols());

    const double eta = params_.eta;
    const auto [sol_fibers, sol_shell] = get_solution_maps(curr_solution_.data());
    const auto &fp = params_.fiber_periphery_interaction;

    Eigen::MatrixXd f_on_fibers = fc_.apply_fiber_force(sol_fibers);
    if (params_.periphery_interaction_flag) {
        int i_fib = 0;
        for (const auto &fib : fc_) {
            f_on_fibers.col(i_fib) += shell_->fiber_interaction(fib, fp);
            i_fib++;
        }
    }

    // clang-format off
    u_trg = fc_.flow(r_trg, f_on_fibers, eta, false) + \
        shell_->flow(r_trg, sol_shell, eta) + \
        psc_.flow(r_trg, eta, properties.time) + \
        bs_.flow(r_trg, eta);
    // clang-format on

    return u_trg;
}

/// @brief Change the pair interaction evaluator method
///
/// @param[in] evaluator (FMM, GPU, CPU)
void set_evaluator(const std::string &evaluator) {
    fc_.set_evaluator(evaluator);
    shell_->set_evaluator(evaluator);
}

/// @brief Calculate all initial velocities/forces/RHS/BCs
///
/// @note Modifies anything that evolves in time.
void prep_state_for_solver() {
    using Eigen::MatrixXd;
    const auto [fib_node_count, shell_node_count] = get_local_node_counts();

    MatrixXd r_all(3, fib_node_count + shell_node_count);
    {
        auto [r_fibers, r_shell] = get_node_maps(r_all);
        r_fibers = fc_.get_local_node_positions();
        r_shell = shell_->get_local_node_positions();
    }

    fc_.update_cache_variables(properties.dt, params_.eta);

    // Implicit motor forces
    MatrixXd motor_force_fibers = params_.implicit_motor_activation_delay > properties.time
                                      ? MatrixXd::Zero(3, fib_node_count)
                                      : fc_.generate_constant_force();

    MatrixXd external_force_fibers = MatrixXd::Zero(3, fib_node_count);
    // Fiber-periphery forces (if periphery exists)
    if (params_.periphery_interaction_flag && shell_->is_active()) {
        int i_col = 0;

        for (const auto &fib : fc_.fibers) {
            external_force_fibers.block(0, i_col, 3, fib.n_nodes_) +=
                shell_->fiber_interaction(fib, params_.fiber_periphery_interaction);
            i_col += fib.n_nodes_;
        }
    }
    // Don't include motor forces for initial calculation (explicitly handled elsewhere)

    MatrixXd v_all = fc_.flow(r_all, external_force_fibers, params_.eta);
    v_all += psc_.flow(r_all, params_.eta, properties.time);
    v_all += bs_.flow(r_all, params_.eta);

    MatrixXd total_force_fibers = motor_force_fibers + external_force_fibers;
    fc_.update_RHS(properties.dt, v_all.block(0, 0, 3, fib_node_count), total_force_fibers);
    fc_.update_boundary_conditions(*shell_, params_.periphery_binding);
    fc_.apply_bc_rectangular(properties.dt, v_all.block(0, 0, 3, fib_node_count), external_force_fibers);

    shell_->update_RHS(v_all.block(0, fib_node_count, 3, shell_node_count));
}

/// @brief Calculate solution vector given current configuration
///
/// @note Modifies too much stuff to note reasonably. RHS info, cache info, and current solution, most notably
/// @return If the solver converged to the requested tolerance with no issue.
bool solve() {
    prep_state_for_solver();

    Solver<P_inv_hydro, A_fiber_hydro> solver_; /// < Wrapper class for solving system
    solver_.set_RHS();

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
    auto [fiber_sol, shell_sol] = get_solution_maps(curr_solution_.data());

    fc_.step(fiber_sol);
    shell_->step(shell_sol);

    return converged;
}

/// @brief store copies of Fiber container in case time step is rejected
void backup() {
    fc_bak_ = fc_;
}

/// @brief restore copies of Fiber container to the state when last backed up
void restore() {
    fc_ = fc_bak_;
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
        double fiber_error = 0.0;
        for (const auto &fib : fc_.fibers) {
            const auto &mats = fib.matrices_.at(fib.n_nodes_);
            const Eigen::MatrixXd xs = std::pow(2.0 / fib.length_, 1) * fib.x_ * mats.D_1_0;
            for (int i = 0; i < fib.n_nodes_; ++i)
                fiber_error = std::max(fabs(xs.col(i).norm() - 1.0), fiber_error);
        }
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
    FiberContainer &fc = fc_;
    Periphery &shell = *shell_;
    const double threshold = 0.0;
    using Eigen::VectorXd;

    char collided = false;
    for (const auto &fiber : fc.fibers)
        if (!collided && shell.check_collision(fiber.x_, threshold))
            collided = true;

    MPI_Allreduce(MPI_IN_PLACE, &collided, 1, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);

    return collided;
}

/// @brief Return copy of fiber container's RHS
Eigen::VectorXd get_fiber_RHS() { return fc_.get_RHS(); }
/// @brief Return copy of shell's RHS
Eigen::VectorXd get_shell_RHS() { return shell_->get_RHS(); }

/// @brief get pointer to params struct
Params *get_params() { return &params_; }
/// @brief get pointer to fiber container
FiberContainer *get_fiber_container() { return &fc_; }
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

    spdlog::info("****** SkellySim {} ({}) ******", SKELLYSIM_VERSION, SKELLYSIM_COMMIT);

    param_table_ = toml::parse(input_file);
    params_ = Params(param_table_.at("params"));
    RNG::init(params_.seed);

    properties.dt = params_.dt_initial;

    if (param_table_.contains("fibers"))
        fc_ = FiberContainer(param_table_.at("fibers").as_array(), params_);
    else
        fc_ = FiberContainer(params_);

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
    if (rank_ == 0)
        ofs_ = std::ofstream(filename, trajectory_open_mode);

    write_config("skelly_sim.initial_config");
}
} // namespace System
