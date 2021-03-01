#define TOML_IMPLEMENTATION
#include <skelly_sim.hpp>

#include <Eigen/Core>
#include <fstream>
#include <random>

#include <parse_util.hpp>
#include <solver_hydro.hpp>
#include <system.hpp>

#include <mpi.h>

// TODO: Refactor all preprocess stuff. It's awful

/// RNG for generating nucleation site positions
std::mt19937 rng;
std::uniform_real_distribution<double> uniform_rng(0.0, 1.0);

/// @brief Convert fiber initial positions/orientations to full coordinate representation
/// @param[out] fiber_array Explicitly instantiated fiber array config
void resolve_fiber_position(toml::table *fiber_table, Eigen::Vector3d &origin) {
    int64_t n_nodes = parse_util::parse_val_key<int64_t>(fiber_table, "n_nodes", -1);
    double length = parse_util::parse_val_key<double>(fiber_table, "length");

    toml::array *x_array = fiber_table->get_as<toml::array>("x");
    toml::array *x_0 = fiber_table->get_as<toml::array>("relative_position");
    toml::array *u = fiber_table->get_as<toml::array>("orientation");

    if (n_nodes == -1) {
        n_nodes = x_array->size() / 3;
        if (n_nodes == 0)
            throw std::runtime_error("Attempt to initialize fiber without point count or positions.");
    }

    if ((!!x_array && !!x_0) || (!!x_array && !!u))
        throw std::runtime_error("Fiber supplied 'relative_position' or 'orientation' with node positions 'x', "
                                 "ambiguous initialization.");

    // FIXME: Make test for this case
    if (!!x_0 && !!u && x_0->size() == 3 && u->size() == 3) {
        fiber_table->insert("x", toml::array());
        x_array = fiber_table->get_as<toml::array>("x");
        Eigen::MatrixXd x(3, n_nodes);
        Eigen::Vector3d rel_pos = parse_util::convert_array<>(x_0);
        Eigen::Vector3d orientation = parse_util::convert_array<>(u);
        Eigen::ArrayXd s = Eigen::ArrayXd::LinSpaced(n_nodes, 0, length).transpose();
        for (int i = 0; i < 3; ++i)
            x.row(i) = origin(i) + rel_pos(i) + orientation(i) * s;
        for (int i = 0; i < x.size(); ++i) {
            x_array->push_back(x.data()[i]);
        }
    }
}

Eigen::Vector3d uniform_on_sphere() {
    const double u = 2 * (uniform_rng(rng)) - 1;
    const double theta = 2 * M_PI * uniform_rng(rng);
    const double factor = sqrt(1 - u * u);
    return Eigen::Vector3d{factor * cos(theta), factor * sin(theta), u};
}

/// @brief Update config to appropriately fill nucleation sites with fibers
/// @param[out] fiber_array Updated fiber array with pointers to nucleation sites
/// @param[out] body_array Updated body array with nucleation sites fixed to current fiber position
void resolve_nucleation_sites(toml::array *fiber_array, toml::array *body_array) {
    using std::make_pair;
    const int n_bodies = body_array->size();
    std::map<std::pair<int, int>, int> occupied;
    std::vector<int> n_sites(n_bodies);

    // Pass through fibers for assigned bodies
    for (size_t i_fib = 0; i_fib < fiber_array->size(); ++i_fib) {
        toml::table *fiber_table = fiber_array->get_as<toml::table>(i_fib);
        int i_body = (*fiber_table)["parent_body"].value_or(-1);
        int i_site = (*fiber_table)["parent_site"].value_or(-1);

        if (i_body >= n_bodies) {
            std::cerr << "Invalid body reference " << i_body << " on fiber " << i_fib << ": \n"
                      << *fiber_table << std::endl;
            throw std::runtime_error("Invalid body reference.\n");
        }

        // unattached or site find automated. move to next fiber
        if (i_body < 0 || i_site < 0)
            continue;

        // check for duplicate assignment
        auto site_pair = make_pair(i_body, i_site);
        if (occupied.count(site_pair) > 0) {
            const int j_fib = occupied[site_pair];
            // clang-format off
                std::cerr << "Multiple fibers bound to site: ("
                          << i_body << ", " << i_site << ")"
                          << " from fibers (" << i_fib << ", " << j_fib << ") " << ".\n"
                          << *fiber_table << "\n\n" << *fiber_array->get_as<toml::table>(j_fib) << "\n";
                throw std::runtime_error("Multiple fibers bound to site.\n");
            // clang-format on
        }

        occupied[site_pair] = i_fib;
    }

    // Pass through fibers for unassigned bodies
    std::vector<int> test_site(n_bodies); //< current site to try for insertion
    for (size_t i_fib = 0; i_fib < fiber_array->size(); ++i_fib) {
        toml::table *fiber_table = fiber_array->get_as<toml::table>(i_fib);
        int i_body = (*fiber_table)["parent_body"].value_or(-1);
        int i_site = (*fiber_table)["parent_site"].value_or(-1);

        // unattached or site already assigned. move to next fiber
        if (i_body < 0 || i_site >= 0)
            continue;

        // Find an empty site
        while (occupied.count(make_pair(i_body, test_site[i_body])) != 0)
            test_site[i_body]++;

        // Store in map so body knows where to grab
        occupied[make_pair(i_body, test_site[i_body])] = i_fib;
    }

    std::vector<std::map<int, Eigen::Vector3d>> nucleation_sites(n_bodies);
    for (const auto &site_map : occupied) {
        auto [i_body, i_site] = site_map.first;
        auto i_fib = site_map.second;
        toml::table *body_table = body_array->get_as<toml::table>(i_body);
        toml::table *fiber_table = fiber_array->get_as<toml::table>(i_fib);
        toml::array *body_position = body_table->get_as<toml::array>("position");

        fiber_table->insert_or_assign("parent_body", i_body);
        fiber_table->insert_or_assign("parent_site", i_site);
        Eigen::Vector3d origin = parse_util::convert_array<>(body_position);

        resolve_fiber_position(fiber_table, origin);

        toml::array *x_array = fiber_table->get_as<toml::array>("x");
        nucleation_sites[i_body][i_site] = {x_array->get_as<double>(0)->get() - origin[0],
                                            x_array->get_as<double>(1)->get() - origin[1],
                                            x_array->get_as<double>(2)->get() - origin[2]};
    }

    for (int i_body = 0; i_body < n_bodies; ++i_body) {
        toml::table *body_table = body_array->get_as<toml::table>(i_body);
        if (!(*body_table)["nucleation_sites"])
            (*body_table).insert("nucleation_sites", toml::array());

        int n_nucleation_sites = (*body_table)["n_nucleation_sites"].value_or(nucleation_sites[i_body].size());
        double radius = *(*body_table)["radius"].value<double>();
        // FIXME: add min_separation parameter
        const double min_separation2 = 0.01 * 0.01;

        for (int i = 0; i < n_nucleation_sites; ++i) {
            if (nucleation_sites[i_body].count(i))
                continue;

            bool collision = false;
            do {
                collision = false;
                nucleation_sites[i_body][i] = radius * uniform_on_sphere();
                Eigen::Vector3d &r_i = nucleation_sites[i_body][i];

                for (auto &j_pair : nucleation_sites[i_body]) {
                    const int &j = j_pair.first;
                    const Eigen::Vector3d &r_j = j_pair.second;
                    if (i == j)
                        continue;
                    Eigen::Vector3d dr = r_j - r_i;
                    double dr2 = dr.dot(dr);
                    if (dr2 < min_separation2) {
                        collision = true;
                        break;
                    }
                }
            } while (collision);
        }

        toml::array *x = body_table->get_as<toml::array>("nucleation_sites");
        for (auto &[site, xvec] : nucleation_sites[i_body])
            for (int i = 0; i < 3; ++i)
                x->push_back(xvec[i]);
    }
}

/// @file
/// @brief Initialize classes with proper interdependencies

/// @brief Patch up input toml config to go from implicit to explicit instantiations
///
/// Some classes have interdependencies, but in order to avoid the objects having explicit
/// dependencies on each other through initialization while maintaining a simple config format,
/// we preprocess the toml config to have TOML communicate these dependencies by explicitly
/// instantiating config values.
///
/// The main example of the utility of this approach is Fiber-Body connections through
/// "nucleation sites". The body nucleation site is _constrained_ to be at the same position as
/// its associated Fiber minus end. Ultimately the Body doesn't care what Fiber it's attached
/// to, just that it has nucleation sites that rotate/move with it. The Fibers, however, have
/// to know which Body they're attached to to constrain their motion. Rather than forcing the
/// user to supply the Fiber minus end and which (body,site) combo it sits at on the Fiber
/// object to the body object, and the nucleation site coordinates in an ordered array on the
/// body object, it's much easier for the user to simply supply a fiber and a body it's attached
/// to, and then automatically sort out the interconnections programmatically.
///
/// This presents an ordering problem for initialization though, since the FiberContainer needs
/// to communicate the nucleation positions to the BodyContainer, but the BodyContainer needs
/// to communicate what (body,site) combo the Fiber is attached to back to the
/// FiberContainer. If you initialize the bodies first, they won't know how to initialize
/// nucleation sites. If you initialize fibers first, they have to communicate the bodies
/// they're attached to the body objects and then get a response on what site they are
/// assigned. This initialization cycle can be removed by just preprocessing the config so that
/// all data on all MPI ranks is consistent at system initialization time, and all objects know
/// the relevant data to fully initialize at object creation time.
///
/// @param[out] config global toml config after initial parsing
/// @param[in] seed seed for RNG state (such as for generating nucleation site positions)
void preprocess(toml::table &config, unsigned long seed) {
    rng.seed(seed);
    toml::array *body_array = config["bodies"].as<toml::array>();
    toml::array *fiber_array = config["fibers"].as<toml::array>();

    // Body-fiber interactions through nucleation sites
    if (!!body_array && !!fiber_array)
        resolve_nucleation_sites(fiber_array, body_array);
    else if (!!fiber_array) {
        for (size_t i_fib = 0; i_fib < fiber_array->size(); ++i_fib) {
            toml::table *fiber_table = fiber_array->get_as<toml::table>(i_fib);
            Eigen::Vector3d origin{0.0, 0.0, 0.0};
            resolve_fiber_position(fiber_table, origin);
        }
    }
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> System::calculate_body_fiber_link_conditions(VectorRef &fibers_xt,
                                                                                         MatrixRef &body_velocities) {
    using Eigen::ArrayXXd;
    using Eigen::MatrixXd;
    using Eigen::Vector3d;

    auto &fc = System::get_fiber_container();
    auto &bc = System::get_body_container();

    Eigen::MatrixXd force_torque_on_bodies = MatrixXd::Zero(6, bc.get_global_count());
    Eigen::MatrixXd velocities_on_fiber = MatrixXd::Zero(7, fc.get_local_count());

    int xt_offset = 0;
    for (size_t i_fib = 0; i_fib < fc.fibers.size(); ++i_fib) {
        const auto &fib = fc.fibers[i_fib];
        const auto &fib_mats = fib.matrices_.at(fib.n_nodes_);
        const int n_pts = fib.n_nodes_;

        auto &[i_body, i_site] = fib.binding_site_;
        if (i_body < 0)
            continue;

        Vector3d site_pos = bc.get_nucleation_site(i_body, i_site) - bc.at(i_body).get_position();
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
        force_torque_on_bodies.col(i_body).segment(0, 3) += F_body;
        force_torque_on_bodies.col(i_body).segment(3, 3) += L_body;

        // SECOND FROM BODY ON-TO FIBER
        // Translational and angular velocities at the attacment point are calculated
        Vector3d v_body = body_velocities.block(0, i_body, 3, 1);
        Vector3d w_body = body_velocities.block(3, i_body, 3, 1);

        // dx/dt = U + Omega x link_loc (move to LHS so -U -Omega x link_loc)
        Vector3d v_fiber = -v_body - w_body.cross(site_pos);

        // tension condition = -(xs*vx + ys*vy + zs*wz)
        double tension_condition = -xs_0.dot(v_body) + (xs_0.cross(site_pos)).dot(w_body);

        // Rotational velocity condition on fiber
        // FIXME: Fiber torque assumes body is a sphere :(
        Vector3d w_fiber = site_pos.normalized().cross(w_body);

        velocities_on_fiber.block(0, i_fib, 3, 1) = v_fiber;
        velocities_on_fiber(3, i_fib) = tension_condition;
        velocities_on_fiber.block(4, i_fib, 3, 1) = w_fiber;

        xt_offset += 4 * n_pts;
    }

    // Sum up fiber contributions from all other ranks to body torques
    MPI_Allreduce(MPI_IN_PLACE, force_torque_on_bodies.data(), force_torque_on_bodies.size(), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    return std::make_pair(force_torque_on_bodies, velocities_on_fiber);
}

std::tuple<int, int, int> get_local_node_counts() {
    return std::make_tuple(System::get_fiber_container().get_local_node_count(),
                           System::get_shell().get_local_node_count(),
                           System::get_body_container().get_local_node_count());
}

std::tuple<int, int, int> System::get_local_solution_sizes() {
    return std::make_tuple(System::get_fiber_container().get_local_solution_size(),
                           System::get_shell().get_local_solution_size(),
                           System::get_body_container().get_local_solution_size());
}

std::tuple<Eigen::Map<Eigen::VectorXd>, Eigen::Map<Eigen::VectorXd>, Eigen::Map<Eigen::VectorXd>>
get_solution_maps(double *x) {
    using Eigen::Map;
    using Eigen::VectorXd;
    auto [fib_sol_size, shell_sol_size, body_sol_size] = System::get_local_solution_sizes();
    return std::make_tuple(VectorMap(x, fib_sol_size), VectorMap(x + fib_sol_size, shell_sol_size),
                           VectorMap(x + fib_sol_size + shell_sol_size, body_sol_size));
}

std::tuple<Eigen::Map<const Eigen::VectorXd>, Eigen::Map<const Eigen::VectorXd>, Eigen::Map<const Eigen::VectorXd>>
get_solution_maps(const double *x) {
    using Eigen::Map;
    using Eigen::VectorXd;
    auto [fib_sol_size, shell_sol_size, body_sol_size] = System::get_local_solution_sizes();
    return std::make_tuple(CVectorMap(x, fib_sol_size), CVectorMap(x + fib_sol_size, shell_sol_size),
                           CVectorMap(x + fib_sol_size + shell_sol_size, body_sol_size));
}

template <typename Derived>
std::tuple<Eigen::Block<Derived>, Eigen::Block<Derived>, Eigen::Block<Derived>>
get_node_maps(Eigen::MatrixBase<Derived> &x) {
    auto [fib_nodes, shell_nodes, body_nodes] = get_local_node_counts();
    return std::make_tuple(Eigen::Block<Derived>(x.derived(), 0, 0, 3, fib_nodes),
                           Eigen::Block<Derived>(x.derived(), 0, fib_nodes, 3, shell_nodes),
                           Eigen::Block<Derived>(x.derived(), 0, fib_nodes + shell_nodes, 3, body_nodes));
}

Eigen::VectorXd System::apply_preconditioner(VectorRef &x) {
    const auto [fib_sol_size, shell_sol_size, body_sol_size] = get_local_solution_sizes();
    const int sol_size = fib_sol_size + shell_sol_size + body_sol_size;
    assert(sol_size == x.size());
    Eigen::VectorXd res(sol_size);

    auto [x_fibers, x_shell, x_bodies] = get_solution_maps(x.data());
    auto [res_fibers, res_shell, res_bodies] = get_solution_maps(res.data());

    res_fibers = System::get_fiber_container().apply_preconditioner(x_fibers);
    res_shell = System::get_shell().apply_preconditioner(x_shell);
    res_bodies = System::get_body_container().apply_preconditioner(x_bodies);

    return res;
}

Eigen::VectorXd System::apply_matvec(VectorRef &x) {
    using Eigen::Block;
    using Eigen::MatrixXd;
    const FiberContainer &fc = System::get_fiber_container();
    const Periphery &shell = System::get_shell();
    const BodyContainer &bc = System::get_body_container();
    const double eta = System::get_params().eta;

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
    r_bodies = bc.get_local_node_positions();

    auto [x_fibers, x_shell, x_bodies] = get_solution_maps(x.data());
    auto [res_fibers, res_shell, res_bodies] = get_solution_maps(res.data());

    // calculate fiber-fiber velocity
    MatrixXd fw = fc.apply_fiber_force(x_fibers);
    Block<MatrixXd> r_shellbody = r_all.block(0, r_fibers.cols(), 3, r_shell.cols() + r_bodies.cols());
    MatrixXd v_fib2all = fc.flow(fw, r_shellbody, eta);

    MatrixXd r_fibbody(3, r_fibers.cols() + r_bodies.cols());
    r_fibbody.block(0, 0, 3, r_fibers.cols()) = r_fibers;
    r_fibbody.block(0, r_fibers.cols(), 3, r_bodies.cols()) = r_bodies;
    MatrixXd v_shell2fibbody = shell.flow(r_fibbody, x_shell, eta);

    MatrixXd body_velocities, body_densities, v_fib_boundary, force_torque_bodies;
    std::tie(body_velocities, body_densities) = bc.unpack_solution_vector(x_bodies);
    std::tie(force_torque_bodies, v_fib_boundary) =
        System::calculate_body_fiber_link_conditions(x_fibers, body_velocities);

    v_all = v_fib2all;
    v_fibers += v_shell2fibbody.block(0, 0, 3, r_fibers.cols());
    v_bodies += v_shell2fibbody.block(0, r_fibers.cols(), 3, r_bodies.cols());
    v_all += bc.flow(r_all, body_densities, force_torque_bodies, eta);

    res_fibers = fc.matvec(x_fibers, v_fibers, v_fib_boundary);
    res_shell = shell.matvec(x_shell, v_shell);
    res_bodies = bc.matvec(v_bodies, body_densities, body_velocities);

    return res;
}

void System::step() {
    using Eigen::MatrixXd;
    Params &params = System::get_params();
    Periphery &shell = System::get_shell();
    FiberContainer &fc = System::get_fiber_container();
    BodyContainer &bc = System::get_body_container();
    const double eta = params.eta;
    const double dt = params.dt;
    const auto [fib_node_count, shell_node_count, body_node_count] = get_local_node_counts();

    MatrixXd r_trg_external(3, shell_node_count + body_node_count);
    r_trg_external.block(0, 0, 3, shell_node_count) = shell.get_local_node_positions();
    r_trg_external.block(0, shell_node_count, 3, body_node_count) = bc.get_local_node_positions();

    fc.update_cache_variables(dt, eta);
    MatrixXd f_on_fibers = fc.generate_constant_force();
    MatrixXd v_fib2all = fc.flow(f_on_fibers, r_trg_external, eta);

    fc.update_RHS(dt, v_fib2all.block(0, 0, 3, fib_node_count), f_on_fibers.block(0, 0, 3, fib_node_count));
    fc.apply_BC_rectangular(dt, v_fib2all.block(0, 0, 3, fib_node_count), f_on_fibers.block(0, 0, 3, fib_node_count));

    shell.update_RHS(v_fib2all.block(0, fib_node_count, 3, shell_node_count));

    bc.update_cache_variables(eta);
    bc.update_RHS(v_fib2all.block(0, fib_node_count + shell_node_count, 3, body_node_count));

    Solver<P_inv_hydro, A_fiber_hydro> solver_;
    solver_.set_RHS();
    solver_.solve();
    CVectorMap sol = solver_.get_solution();

    double residual = solver_.get_residual();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        std::cout << "Residual: " << residual << std::endl;

    auto [fiber_sol, shell_sol, body_sol] = get_solution_maps(sol.data());

    size_t offset = 0;
    for (auto &fib : fc.fibers) {
        for (int i = 0; i < 3; ++i)
            fib.x_.row(i) = sol.segment(offset + i * fib.n_nodes_, fib.n_nodes_);
        offset += 4 * fib.n_nodes_;
    }

    Eigen::MatrixXd body_velocities, body_densities;
    std::tie(body_velocities, body_densities) = bc.unpack_solution_vector(body_sol);

    for (int i = 0; i < bc.bodies.size(); ++i) {
        auto &body = bc.bodies[i];
        Eigen::Vector3d x_new = body.position_ + body_velocities.col(i).segment(0, 3) * dt;
        Eigen::Vector3d phi = body_velocities.col(i).segment(3, 3) * dt;
        double phi_norm = phi.norm();
        if (phi_norm) {
            double s = std::cos(0.5 * phi_norm);
            Eigen::Vector3d p = std::sin(0.5 * phi_norm) * phi / phi_norm;
            Eigen::Quaterniond orientation_new = Eigen::Quaterniond(s, p[0], p[1], p[2]) * body.orientation_;
            body.move(x_new, orientation_new);
        }
    }
}

System::System(std::string *input_file) {
    if (input_file == nullptr)
        throw std::runtime_error("System uninitialized. Call System::init(\"config_file\").");

    param_table_ = toml::parse_file(*input_file);
    params_ = Params(param_table_.get_as<toml::table>("params"));
    preprocess(param_table_, params_.seed);

    fc_ = FiberContainer(param_table_.get_as<toml::array>("fibers"), params_);
    shell_ = params_.shell_precompute_file.length() ? Periphery(params_.shell_precompute_file) : Periphery();
    bc_ = BodyContainer(param_table_.get_as<toml::array>("bodies"), params_);
}
