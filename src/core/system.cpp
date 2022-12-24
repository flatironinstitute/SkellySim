#include <skelly_sim.hpp>

#include <csignal>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>

#include <body.hpp>
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

#include <cnpy.hpp>

#include <mpi.h>
#include <sys/mman.h>

#include <spdlog/cfg/env.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace System {
Params params_;                    ///< Simulation input parameters
FiberContainer fc_;                ///< Fibers
BodyContainer bc_;                 ///< Bodies
PointSourceContainer psc_;         ///< Point Sources
std::unique_ptr<Periphery> shell_; ///< Periphery
std::vector<PointSource> points_;  ///< External point sources
struct properties_t properties {
    .dt = 0.0, .time = 0.0,
};

struct properties_t &get_properties() {
    return properties;
};

Eigen::VectorXd curr_solution_;

Eigen::VectorXd &get_curr_solution() { return curr_solution_; }

std::ofstream ofs_;    ///< Trajectory output file stream. Opened at initialization
std::ofstream ofs_vf_; ///< Velocity field output file stream. Opened at initialization

FiberContainer fc_bak_;   ///< Copy of fibers for timestep reversion
BodyContainer bc_bak_;    ///< Copy of bodies for timestep reversion
int rank_;                ///< MPI rank
int size_;                ///< MPI size
toml::value param_table_; ///< Parsed input table
bool resume_flag_;        ///< FIXME: Hack check if resuming or post-processing/initial run

/// @brief Get number of physical nodes local to MPI rank for each object type [fibers, shell, bodies]
std::tuple<int, int, int>
get_local_node_counts() {
    return std::make_tuple(fc_.get_local_node_count(), shell_->get_local_node_count(), bc_.get_local_node_count());
}

/// @brief Get GMRES solution size local to MPI rank for each object type [fibers, shell, bodies]
std::tuple<int, int, int> get_local_solution_sizes() {
    return std::make_tuple(fc_.get_local_solution_size(), shell_->get_local_solution_size(),
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

/// @brief Flush current simulation state to trajectory file(s)
void write(std::ofstream &ofs) {
    FiberContainer fc_global;
    BodyContainer bc_empty;
    BodyContainer &bc_global = (rank_ == 0) ? bc_ : bc_empty;
    Periphery shell_global;

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

void write() { write(ofs_); }

void write_config(const std::string &config_file) {
    auto trajectory_open_mode = std::ofstream::binary | std::ofstream::out;
    auto ofs = std::ofstream(config_file, trajectory_open_mode);
    write(ofs);
}

void interrupt_handler(int signum) {
    restore();
    write_config("skelly_sim.final_config");
    throw std::runtime_error("Manual interrupt");
}

/// @brief Class representing a velocity field
/// This allows for trivial dumping of the VF 'trajectory'
class VelocityField {
  public:
    double time;            ///< Current time
    Eigen::MatrixXd x_grid; ///< [3 x n_grid_points] matrix of points to evaluate the field
    Eigen::MatrixXd v_grid; ///< [3 x n_grid_points] matrix of velocities at x_grid
    void compute();         ///< Compute the velocity field given the current system configuration, and x_
    Eigen::MatrixXd make_grid();
    void write() { ///< Flush the velocity field to disk
        int total_count = displs_.back();
        Eigen::MatrixXd x_grid_tot;
        Eigen::MatrixXd v_grid_tot;
        if (rank_ == 0) {
            x_grid_tot.resize(3, total_count / 3);
            v_grid_tot.resize(3, total_count / 3);
            spdlog::debug("I'm here... {} {} {}", counts_[rank_], total_count, displs_.back());
        }

        MPI_Gatherv(x_grid.data(), counts_[rank_], MPI_DOUBLE, x_grid_tot.data(), counts_.data(), displs_.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(v_grid.data(), counts_[rank_], MPI_DOUBLE, v_grid_tot.data(), counts_.data(), displs_.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        x_grid = x_grid_tot;
        v_grid = v_grid_tot;

        msgpack::pack(ofs_vf_, *this);
        ofs_vf_.flush();
    };
    MSGPACK_DEFINE_MAP(time, x_grid, v_grid);

  private:
    std::vector<int> counts_;
    std::vector<int> displs_;
};

/// @brief Set system state to last state found in trajectory files
///
/// @param[in] input_file input file name of trajectory file for this rank
void resume_from_trajectory(std::string input_file) {
    TrajectoryReader trajectory(input_file, true);
    while (trajectory.read_next_frame()) {
    }
    trajectory.unpack_current_frame();
}

// /// @brief Generate uniformly distributed point on unit sphere
// Eigen::Vector3d uniform_on_sphere() {
//     const double u = 2 * (RNG::uniform_unsplit()) - 1;
//     const double theta = 2 * M_PI * RNG::uniform_unsplit();
//     const double factor = sqrt(1 - u * u);
//     return Eigen::Vector3d{factor * cos(theta), factor * sin(theta), u};
// }

/// @brief Calculate forces/torques on the bodies and velocities on the fibers due to attachment constraints
/// @param[in] fibers_xt [4 x num_fiber_nodes_local] Vector of fiber node positions and tensions on current rank.
/// Ordering is [fib1.nodes.x, fib1.nodes.y, fib1.nodes.z, fib1.T, fib2.nodes.x, ...]
/// @param[in] x_bodies entire body component of the solution vector (deformable+rigid)
Eigen::MatrixXd calculate_body_fiber_link_conditions(VectorRef &fibers_xt, VectorRef &x_bodies) {
    using Eigen::ArrayXXd;
    using Eigen::MatrixXd;
    using Eigen::Vector3d;

    auto &fc = fc_;
    auto &bc = bc_;

    Eigen::MatrixXd velocities_on_fiber = MatrixXd::Zero(7, fc.get_local_count());

    MatrixXd body_velocities(6, bc.spherical_bodies.size());
    int index = 0;
    for (const auto &body : bc.spherical_bodies) {
        body_velocities.col(index) =
            x_bodies.segment(bc.solution_offsets_.at(std::static_pointer_cast<Body>(body)) + body->n_nodes_ * 3, 6);
    }

    int xt_offset = 0;
    int i_fib = 0;
    for (auto &body : bc.spherical_bodies)
        body->force_torque_.setZero();

    for (const auto &fib : fc.fibers) {
        const auto &fib_mats = fib.matrices_.at(fib.n_nodes_);
        const int n_pts = fib.n_nodes_;

        auto &[i_body, i_site] = fib.binding_site_;
        if (i_body < 0)
            continue;

        auto body = std::static_pointer_cast<SphericalBody>(bc.bodies[i_body]);
        Vector3d site_pos = bc.get_nucleation_site(i_body, i_site) - body->get_position();
        MatrixXd x_new(3, fib.n_nodes_);
        x_new.row(0) = fibers_xt.segment(xt_offset + 0 * fib.n_nodes_, fib.n_nodes_);
        x_new.row(1) = fibers_xt.segment(xt_offset + 1 * fib.n_nodes_, fib.n_nodes_);
        x_new.row(2) = fibers_xt.segment(xt_offset + 2 * fib.n_nodes_, fib.n_nodes_);

        double T_new_0 = fibers_xt(xt_offset + 3 * n_pts);

        Vector3d xs_0 = fib.xs_.col(0);
        Eigen::MatrixXd xss_new = pow(2.0 / fib.length_, 2) * x_new * fib_mats.D_2_0;
        Eigen::MatrixXd xsss_new = pow(2.0 / fib.length_, 3) * x_new * fib_mats.D_3_0;
        Vector3d xss_new_0 = xss_new.col(0);
        Vector3d xsss_new_0 = xsss_new.col(0);

        // FIRST FROM FIBER ON-TO BODY
        // Force by fiber on body at s = 0, Fext = -F|s=0 = -(EXsss - TXs)
        // Bending term + Tension term:
        Vector3d F_body = -fib.bending_rigidity_ * xsss_new_0 + xs_0 * T_new_0;

        // Torque by fiber on body at s = 0
        // Lext = (L + link_loc x F) = -E(Xss x Xs) - link_loc x (EXsss - TXs)
        // bending contribution :
        Vector3d L_body = -fib.bending_rigidity_ * site_pos.cross(xsss_new_0);

        // tension contribution :
        L_body += site_pos.cross(xs_0) * T_new_0;

        // fiber's torque L:
        L_body += fib.bending_rigidity_ * xs_0.cross(xss_new_0);

        // Store the contribution of each fiber in this array
        body->force_torque_.segment(0, 3) += F_body;
        body->force_torque_.segment(3, 3) += L_body;

        // SECOND FROM BODY ON-TO FIBER
        // Translational and angular velocities at the attachment point are calculated
        Vector3d v_body = body_velocities.block(0, i_body, 3, 1);
        Vector3d w_body = body_velocities.block(3, i_body, 3, 1);

        // dx/dt = U + Omega x link_loc (move to LHS so -U -Omega x link_loc)
        Vector3d v_fiber = -v_body - w_body.cross(site_pos);

        // tension condition = -(xs*vx + ys*vy + zs*wz)
        double tension_condition = -xs_0.dot(v_body) + (xs_0.cross(site_pos)).dot(w_body);

        // Rotational velocity condition on fiber
        // FIXME: Fiber torque assumes body is a sphere :(
        Vector3d w_fiber = site_pos.normalized().cross(w_body);

        velocities_on_fiber.col(i_fib).segment(0, 3) = v_fiber;
        velocities_on_fiber(3, i_fib) = tension_condition;
        velocities_on_fiber.col(i_fib).segment(4, 3) = w_fiber;

        i_fib++;
        xt_offset += 4 * n_pts;
    }

    return velocities_on_fiber;
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
Eigen::VectorXd apply_preconditioner(VectorRef &x) {
    const auto [fib_sol_size, shell_sol_size, body_sol_size] = get_local_solution_sizes();
    const int sol_size = fib_sol_size + shell_sol_size + body_sol_size;
    assert(sol_size == x.size());
    Eigen::VectorXd res(sol_size);

    auto [x_fibers, x_shell, x_bodies] = get_solution_maps(x.data());
    auto [res_fibers, res_shell, res_bodies] = get_solution_maps(res.data());

    res_fibers = fc_.apply_preconditioner(x_fibers);
    res_shell = shell_->apply_preconditioner(x_shell);
    res_bodies = bc_.apply_preconditioner(x_bodies);

    return res;
}

/// @brief Apply and return entire operator on system state vector for fibers/body/shell
///
/// \f[ A * x = y \f]
/// @param [in] x [local_solution_size] Vector to apply matvec on
/// @return [local_solution_size] Vector y, the result of the operator applied to x.
Eigen::VectorXd apply_matvec(VectorRef &x) {
    using Eigen::Block;
    using Eigen::MatrixXd;
    const FiberContainer &fc = fc_;
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
    MatrixXd v_fib_boundary = System::calculate_body_fiber_link_conditions(x_fibers, x_bodies_global);

    v_all = v_fib2all;
    v_fibers += v_shell2fibbody.block(0, 0, 3, r_fibers.cols());
    v_bodies += v_shell2fibbody.block(0, r_fibers.cols(), 3, r_bodies.cols());
    v_all += bc.flow(r_all, x_bodies, eta);

    res_fibers = fc.matvec(x_fibers, v_fibers, v_fib_boundary);
    res_shell = shell.matvec(x_shell, v_shell);
    res_bodies = bc.matvec(v_bodies, x_bodies);

    return res;
}

/// @brief Create 'grid' for velocity field base on the vf parameters. Remove points from the
/// grid where they lie outside the periphery or repeat (if multiple bodies and a 'moving'
/// grid)
///
/// @return 3D list of points defining the grid
Eigen::MatrixXd VelocityField::make_grid() {
    const auto &vf = params_.velocity_field;
    Eigen::MatrixXd grid_master;
    std::unordered_map<long int, Eigen::Vector3d> grid_map;
    using Vector3l = Eigen::Matrix<long int, 3, 1>;
    const Vector3l key_map(1, 100000, 10000000000);

    if (rank_ == 0) {
        const double res = vf.resolution;
        if (vf.moving_volume) {
            Eigen::MatrixXd sphere_centers = bc_.get_global_center_positions(bc_.spherical_bodies);
            int n_points = 1 + (2.0 * vf.moving_volume_radius / res);

            for (int i_grid = 0; i_grid < sphere_centers.cols(); ++i_grid) {
                Vector3l bottom_left =
                    ((sphere_centers.col(i_grid).array() - vf.moving_volume_radius) / res).floor().cast<long int>();
                for (int i = 0; i < n_points; ++i) {
                    for (int j = 0; j < n_points; ++j) {
                        for (int k = 0; k < n_points; ++k) {
                            Vector3l coord_i = Vector3l{i, j, k} + bottom_left;
                            long int key = key_map[0] * coord_i[0] + key_map[1] * coord_i[1] + key_map[2] * coord_i[2];
                            Eigen::Vector3d test_point = res * coord_i.cast<double>();

                            if (!shell_->check_collision(test_point, 0.0))
                                grid_map[key] = test_point;
                        }
                    }
                }
            }
            grid_master.resize(3, grid_map.size());
            int pos = 0;
            for (const auto &point : grid_map) {
                grid_master.col(pos) = point.second;
                pos++;
            }
        } else {
            auto [a, b, c] = shell_->get_dimensions();
            int n_points_x = 1 + 2.0 * a / res;
            int n_points_y = 1 + 2.0 * b / res;
            int n_points_z = 1 + 2.0 * c / res;
            grid_master.resize(3, n_points_x * n_points_y * n_points_z);

            int i_col = 0;
            for (int i = 0; i < n_points_x; ++i) {
                for (int j = 0; j < n_points_y; ++j) {
                    for (int k = 0; k < n_points_z; ++k) {
                        Eigen::Vector3d x{-a + i * res, -b + j * res, -c + k * res};
                        if (!shell_->check_collision(x, 0.0)) {
                            grid_master.col(i_col) = x;
                            i_col++;
                        }
                    }
                }
            }
            grid_master.conservativeResize(3, i_col);
        }
    }

    Eigen::MatrixXd grid;
    long int grid_size_global = grid_master.cols();
    MPI_Bcast(&grid_size_global, 1, MPI_LONG_INT, 0, MPI_COMM_WORLD);

    const int node_size_big = 3 * (grid_size_global / size_ + 1);
    const int node_size_small = 3 * (grid_size_global / size_);
    const int n_nodes_big = grid_size_global % size_;

    counts_.resize(size_);
    displs_.resize(size_ + 1);
    for (int i = 0; i < size_; ++i) {
        counts_[i] = ((i < n_nodes_big) ? node_size_big : node_size_small);
        displs_[i + 1] = displs_[i] + counts_[i];
    }

    grid.resize(3, counts_[rank_] / 3);
    MPI_Scatterv(grid_master.data(), counts_.data(), displs_.data(), MPI_DOUBLE, grid.data(), counts_[rank_],
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return grid;
}

Eigen::MatrixXd velocity_at_targets(MatrixRef &r_trg) {
    if (!r_trg.size())
        return Eigen::MatrixXd(3, 0);
    Eigen::MatrixXd u_trg(r_trg.rows(), r_trg.cols());

    const double eta = params_.eta;
    const auto [sol_fibers, sol_shell, sol_bodies] = get_solution_maps(curr_solution_.data());
    const auto &fp = params_.fiber_periphery_interaction;

    Eigen::MatrixXd f_on_fibers = fc_.apply_fiber_force(sol_fibers);
    if (params_.periphery_interaction_flag) {
        int i_fib = 0;
        for (const auto &fib : fc_) {
            f_on_fibers.col(i_fib) += shell_->fiber_interaction(fib, fp);
            i_fib++;
        }
    }

    // FIXME: This is likely wrong, but more right than before
    Eigen::VectorXd sol_bodies_global(bc_.get_global_solution_size());
    if (rank_ == 0)
        sol_bodies_global = sol_bodies;
    MPI_Bcast(sol_bodies_global.data(), sol_bodies_global.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // This routine zeros out the external force. is that correct?
    calculate_body_fiber_link_conditions(sol_fibers, sol_bodies_global);
    for (auto &body : bc_.spherical_bodies)
        body->force_torque_.segment(0, 3) += body->external_force_;

    // clang-format off
    u_trg = fc_.flow(r_trg, f_on_fibers, eta, false) + \
        bc_.flow(r_trg, sol_bodies, eta) + \
        shell_->flow(r_trg, sol_shell, eta) + \
        psc_.flow(r_trg, eta, properties.time);
    // clang-format on

    // FIXME: move this to body logic with overloading
    // Replace points inside a body to have velocity v_body + w_body x r
    for (int i = 0; i < r_trg.cols(); ++i) {
        for (auto &body : bc_.spherical_bodies) {
            Eigen::Vector3d dx = r_trg.col(i) - body->position_;
            if (dx.norm() < body->radius_)
                u_trg.col(i) = body->velocity_ + body->angular_velocity_.cross(dx);
        }
    }

    return u_trg;
}

void set_evaluator(const std::string &evaluator) {
    fc_.set_evaluator(evaluator);
    bc_.set_evaluator(evaluator);
    shell_->set_evaluator(evaluator);
}

void VelocityField::compute() {
    time = properties.time;
    x_grid = make_grid();
    v_grid = velocity_at_targets(x_grid);
}

/// @brief Calculate all initial velocities/forces/RHS/BCs
///
/// @note Modifies anything that evolves in time.
void prep_state_for_solver() {
    using Eigen::MatrixXd;

    // Since DI can change size of fiber containers, must call first.
    System::dynamic_instability();

    const auto [fib_node_count, shell_node_count, body_node_count] = get_local_node_counts();

    MatrixXd r_all(3, fib_node_count + shell_node_count + body_node_count);
    {
        auto [r_fibers, r_shell, r_bodies] = get_node_maps(r_all);
        r_fibers = fc_.get_local_node_positions();
        r_shell = shell_->get_local_node_positions();
        r_bodies = bc_.get_local_node_positions(bc_.bodies);
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

    bc_.update_cache_variables(params_.eta);

    // Check for an add external body forces
    bool external_force_body = false;
    for (auto &body : bc_.spherical_bodies) {
        body->force_torque_.setZero();
        // Hack so that when you sum global forces, it should sum back to the external force
        body->force_torque_.segment(0, 3) = body->external_force_ / size_;
        external_force_body = external_force_body | body->external_force_.any();
    }

    if (external_force_body) {
        const int total_node_count = fib_node_count + shell_node_count + body_node_count;
        MatrixXd r_all(3, total_node_count);
        auto [r_fibers, r_shell, r_bodies] = get_node_maps(r_all);
        r_fibers = fc_.get_local_node_positions();
        r_shell = shell_->get_local_node_positions();
        r_bodies = bc_.get_local_node_positions(bc_.bodies);

        v_all += bc_.flow(r_all, Eigen::VectorXd::Zero(bc_.get_local_solution_size()), params_.eta);
    }

    v_all += psc_.flow(r_all, params_.eta, properties.time);

    bc_.update_RHS(v_all.block(0, fib_node_count + shell_node_count, 3, body_node_count));

    MatrixXd total_force_fibers = motor_force_fibers + external_force_fibers;
    fc_.update_RHS(properties.dt, v_all.block(0, 0, 3, fib_node_count), total_force_fibers);
    fc_.update_boundary_conditions(*shell_, params_.periphery_binding);
    fc_.apply_bc_rectangular(properties.dt, v_all.block(0, 0, 3, fib_node_count), external_force_fibers);

    shell_->update_RHS(v_all.block(0, fib_node_count, 3, shell_node_count));
}

void dump_get_system_matvec(const std::string &matfile = "system_matrix.mat") {
    int n_cols = get_local_solution_size();
    Eigen::MatrixXd mat(n_cols, n_cols);

    for (int i = 0; i < n_cols; ++i) {
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n_cols);
        b[i] = 1.0;

        // Grab row because numpy is row-major and eigen column-major, so we avoid the
        // necessary transpose
        mat.row(i) = apply_matvec(b);
    }

    cnpy::npy_save(matfile, mat.data(), {(unsigned long)mat.rows(), (unsigned long)mat.cols()}, "w");
}

/// @brief Generate next trial system state for the current System::properties::dt
///
/// @note Modifies anything that evolves in time.
/// @return If the Matrix solver converged to the requested tolerance with no issue.
bool step() {
    const double dt = properties.dt;

    prep_state_for_solver();

    Solver<P_inv_hydro, A_fiber_hydro> solver_; /// < Wrapper class for solving system

    solver_.set_RHS();

    bool converged = solver_.solve();
    curr_solution_ = solver_.get_solution();

    double residual = solver_.get_residual();
    spdlog::info("Residual: {}", residual);

    auto [fiber_sol, shell_sol, body_sol] = get_solution_maps(curr_solution_.data());

    fc_.step(fiber_sol);
    bc_.step(body_sol, dt);
    fc_.repin_to_bodies(bc_);
    shell_->step(shell_sol);

    return converged;
}

/// @brief store copies of Fiber and Body containers in case time step is rejected
void backup() {
    fc_bak_ = fc_;
    bc_bak_ = bc_;
}

/// @brief restore copies of Fiber and Body containers to the state when last backed up
void restore() {
    fc_ = fc_bak_;
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

/// @brief Run the post processing step
void run_post_process() {
    TrajectoryReader trajectory("skelly_sim.out", resume_flag_);

    double dt_write_field = params_.velocity_field.dt_write_field;
    double t_last = -dt_write_field;
    while (trajectory.read_next_frame()) {
        trajectory.unpack_current_frame(true);
        if ((properties.time - t_last) / dt_write_field >= 1.0) {

            VelocityField vf_curr;
            vf_curr.compute();
            vf_curr.write();
            spdlog::info("{}", properties.time);
            t_last = properties.time;
        }
    }
}

/// @brief Check for any collisions between objects
///
/// @return true if any collision detected, false otherwise
bool check_collision() {
    BodyContainer &bc = bc_;
    FiberContainer &fc = fc_;
    Periphery &shell = *shell_;
    const double threshold = 0.0;
    using Eigen::VectorXd;

    char collided = false;
    for (const auto &body : bc.bodies)
        if (!collided && body->check_collision(shell, threshold))
            collided = true;

    for (const auto &fiber : fc.fibers)
        if (!collided && shell.check_collision(fiber.x_, threshold))
            collided = true;

    for (auto &body1 : bc.bodies)
        for (auto &body2 : bc.bodies)
            if (!collided && body1 != body2 && body1->check_collision(*body2, threshold))
                collided = true;

    MPI_Allreduce(MPI_IN_PLACE, &collided, 1, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);

    return collided;
}

/// @brief Return copy of fiber container's RHS
Eigen::VectorXd get_fiber_RHS() { return fc_.get_RHS(); }
/// @brief Return copy of body container's RHS
Eigen::VectorXd get_body_RHS() { return bc_.get_RHS(); }
/// @brief Return copy of shell's RHS
Eigen::VectorXd get_shell_RHS() { return shell_->get_RHS(); }

/// @brief get pointer to params struct
Params *get_params() { return &params_; }
/// @brief get pointer to body container
BodyContainer *get_body_container() { return &bc_; }
/// @brief get pointer to fiber container
FiberContainer *get_fiber_container() { return &fc_; }
/// @brief get pointer to shell
Periphery *get_shell() { return shell_.get(); }
/// @brief get pointer to param table struct
toml::value *get_param_table() { return &param_table_; }

/// @brief Initialize entire system. Needs to be called once at the beginning of the program execution
/// @param[in] input_file String of toml config file specifying system parameters and initial conditions
/// @param[in] resume_flag true if simulation is resuming from prior execution state, false otherwise.
void init(const std::string &input_file, bool resume_flag, bool post_process_flag, bool listen_flag) {
    resume_flag_ = resume_flag;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    signal(SIGINT, interrupt_handler);
    if (listen_flag) {
        spdlog::logger sink =
            rank_ == 0 ? spdlog::logger("SkellySim", std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>())
                       : spdlog::logger("SkellySim", std::make_shared<spdlog::sinks::null_sink_st>());
        spdlog::set_default_logger(std::make_shared<spdlog::logger>(sink));
        spdlog::stderr_color_mt("STKFMM");
        spdlog::stderr_color_mt("Belos");
        spdlog::stderr_color_mt("SkellySim global");
    } else {
        spdlog::logger sink =
            rank_ == 0 ? spdlog::logger("SkellySim", std::make_shared<spdlog::sinks::ansicolor_stdout_sink_st>())
                       : spdlog::logger("SkellySim", std::make_shared<spdlog::sinks::null_sink_st>());
        spdlog::set_default_logger(std::make_shared<spdlog::logger>(sink));
        spdlog::stdout_color_mt("STKFMM");
        spdlog::stdout_color_mt("Belos");
        spdlog::stdout_color_mt("SkellySim global");
    }
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

    if (param_table_.contains("bodies"))
        bc_ = BodyContainer(param_table_.at("bodies").as_array(), params_);

    if (param_table_.contains("point_sources"))
        psc_ = PointSourceContainer(param_table_.at("point_sources").as_array());

    curr_solution_.resize(get_local_solution_size());
    std::string filename = "skelly_sim.out";
    auto trajectory_open_mode =
        std::ofstream::binary | ((post_process_flag || listen_flag) ? std::ofstream::in : std::ofstream::out);
    if (resume_flag) {
        resume_from_trajectory(filename);
        trajectory_open_mode = trajectory_open_mode | std::ofstream::app;
    }
    if (rank_ == 0)
        ofs_ = std::ofstream(filename, trajectory_open_mode);

    if (post_process_flag && rank_ == 0)
        ofs_vf_ = std::ofstream("skelly_sim.vf", std::ofstream::binary | std::ofstream::out);

    write_config("skelly_sim.initial_config");
}
} // namespace System
