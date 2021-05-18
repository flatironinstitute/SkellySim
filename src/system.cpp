#include <skelly_sim.hpp>

#include <rng.hpp>

#include <Eigen/Core>
#include <fstream>
#include <unordered_map>

#include <body.hpp>
#include <fiber.hpp>
#include <params.hpp>
#include <parse_util.hpp>
#include <periphery.hpp>
#include <solver_hydro.hpp>
#include <system.hpp>

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
std::unique_ptr<Periphery> shell_; ///< Periphery
std::ofstream ofs_;                ///< Trajectory output file stream. Opened at initialization

FiberContainer fc_bak_;   ///< Copy of fibers for timestep reversion
BodyContainer bc_bak_;    ///< Copy of bodies for timestep reversion
int rank_;                ///< MPI rank
int size_;                ///< MPI size
toml::value param_table_; ///< Parsed input table

/// @brief Time varying system properties that are extrinsic to the physical objects
struct {
    double dt;         ///< Current timestep size
    double time = 0.0; ///< Current system time
} properties;

/// @brief Structure for trajectory output via msgpack
///
/// This can be extended easily, so long as you update the corresponding input_map_t and potentially the System::write()
/// function if you can't use a reference in the output_map for some reason.
typedef struct output_map_t {
    double &time = properties.time;                          ///< System::properties
    double &dt = properties.dt;                              ///< System::properties
    FiberContainer &fibers = fc_;                            ///< System::fc_
    BodyContainer &bodies = bc_;                             ///< System::bc_
    std::pair<std::string, std::string> rng_state;           ///< string representation of split/unsplit state in RNG
    MSGPACK_DEFINE_MAP(time, dt, rng_state, fibers, bodies); ///< Helper routine to specify serialization
} output_map_t;
output_map_t output_map; ///< Output data for msgpack dump

/// @brief Structure for importing frame of trajectory into the simulation
///
/// We can't use output_map_t here, but rather a similar struct which uses copies of the member
/// variables (rather than references) which are then used to update the System variables.
typedef struct input_map_t {
    double time;                                             ///< System::properties
    double dt;                                               ///< System::properties
    FiberContainer fibers;                                   ///< System::fc_
    BodyContainer bodies;                                    ///< System::bc_
    std::pair<std::string, std::string> rng_state;           ///< string representation of split/unsplit state in RNG
    MSGPACK_DEFINE_MAP(time, dt, rng_state, fibers, bodies); ///< Helper routine to specify serialization
} input_map_t;

/// Flush current simulation state to trajectory file.
void write() {
    output_map.rng_state = RNG::dump_state();
    msgpack::pack(ofs_, output_map);
    ofs_.flush();
}

/// @brief Set system state to last state found in trajectory files
///
/// @param[in] if_file input file name of trajectory file for this rank
void resume_from_trajectory(std::string if_file) {
    int fd = open(if_file.c_str(), O_RDONLY);
    if (fd == -1)
        throw std::runtime_error("Unable to open trajectory file " + if_file + " for resume.");

    struct stat sb;
    if (fstat(fd, &sb) == -1)
        throw std::runtime_error("Error statting " + if_file + " for resume.");

    std::size_t buflen = sb.st_size;

    const char *addr = static_cast<const char *>(mmap(NULL, buflen, PROT_READ, MAP_PRIVATE, fd, 0u));
    if (addr == MAP_FAILED)
        throw std::runtime_error("Error mapping " + if_file + " for resume.");

    std::size_t offset = 0;
    msgpack::object_handle oh;
    // FIXME: There is probably a way to not have to read the entire trajectory
    while (offset != buflen)
        msgpack::unpack(oh, addr, buflen, offset);

    // FIXME: add assertion that system time is the same across all ranks to resume functionality

    // FIXME: This is a bug-prone way to unpack. Should make proper clone functions that merge
    // the minimum representation with existing data automatically (node data, etc)
    msgpack::object obj = oh.get();
    input_map_t const &min_state = obj.as<input_map_t>();
    output_map.time = min_state.time;
    output_map.dt = min_state.dt;
    fc_.fibers.clear();
    for (const auto &min_fib : min_state.fibers.fibers) {
        Fiber new_fib = min_fib;
        new_fib.init(params_.eta);
        new_fib.x_ = min_fib.x_;
        fc_.fibers.push_back(new_fib);
    }

    // make sure sublist pointers are initialized, and then fill them in
    bc_.populate_sublists();
    for (int i = 0; i < bc_.spherical_bodies.size(); ++i)
        bc_.spherical_bodies[i]->min_copy(min_state.bodies.spherical_bodies[i]);
    for (int i = 0; i < bc_.deformable_bodies.size(); ++i)
        bc_.deformable_bodies[i]->min_copy(min_state.bodies.deformable_bodies[i]);
    output_map.rng_state = min_state.rng_state;
    RNG::init(output_map.rng_state);
}

// TODO: Refactor all preprocess stuff. It's awful

/// @brief Convert fiber initial positions/orientations to full coordinate representation
/// @param[out] fiber_table Explicitly instantiated element of fiber config
/// @param[in] origin origin of coordinate system for fiber
void resolve_fiber_position(toml::value &fiber_table, Eigen::Vector3d &origin) {
    int64_t n_nodes = toml::find_or<int64_t>(fiber_table, "n_nodes", -1);
    double length = toml::find<double>(fiber_table, "length");

    Eigen::VectorXd x_array = parse_util::convert_array<>(toml::find_or<toml::array>(fiber_table, "x", {}));
    Eigen::VectorXd x_0 = parse_util::convert_array<>(toml::find_or<toml::array>(fiber_table, "relative_position", {}));
    Eigen::VectorXd u = parse_util::convert_array<>(toml::find_or<toml::array>(fiber_table, "orientation", {}));

    if (n_nodes == -1) {
        n_nodes = x_array.size() / 3;
        if (n_nodes == 0)
            throw std::runtime_error("Attempt to initialize fiber without point count or positions.");
    }

    if ((x_array.size() && x_0.size()) || (x_array.size() && u.size()))
        throw std::runtime_error("Fiber supplied 'relative_position' or 'orientation' with node positions 'x', "
                                 "ambiguous initialization.");

    // FIXME: Make test for this case
    if (x_0.size() == 3 && u.size() == 3) {
        Eigen::MatrixXd x(3, n_nodes);
        Eigen::ArrayXd s = Eigen::ArrayXd::LinSpaced(n_nodes, 0, length).transpose();
        for (int i = 0; i < 3; ++i)
            x.row(i) = origin(i) + x_0(i) + u(i) * s;

        fiber_table.as_table()["x"] = toml::array();
        toml::array &x_tab = fiber_table.at("x").as_array();
        for (int i = 0; i < x.size(); ++i) {
            x_tab.push_back(x.data()[i]);
        }
    }
}

/// @brief Generate uniformly distributed point on unit sphere
Eigen::Vector3d uniform_on_sphere() {
    const double u = 2 * (RNG::uniform_unsplit()) - 1;
    const double theta = 2 * M_PI * RNG::uniform_unsplit();
    const double factor = sqrt(1 - u * u);
    return Eigen::Vector3d{factor * cos(theta), factor * sin(theta), u};
}

/// @brief Update config to appropriately fill nucleation sites with fibers
/// @param[out] fiber_array Updated fiber array with pointers to nucleation sites
/// @param[out] body_array Updated body array with nucleation sites fixed to current fiber position
void resolve_nucleation_sites(toml::array &fiber_array, toml::array &body_array) {
    using std::make_pair;
    const int n_bodies = body_array.size();
    std::map<std::pair<int, int>, int> occupied;
    std::vector<int> n_sites(n_bodies);

    // Pass through fibers for assigned bodies
    for (size_t i_fib = 0; i_fib < fiber_array.size(); ++i_fib) {
        toml::value &fiber_table = fiber_array.at(i_fib);
        int i_body = toml::find_or<int>(fiber_table, "parent_body", -1);
        int i_site = toml::find_or<int>(fiber_table, "parent_site", -1);

        if (i_body >= n_bodies) {
            std::cerr << "Invalid body reference " << i_body << " on fiber " << i_fib << ": \n"
                      << fiber_table << std::endl;
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
                          << fiber_table << "\n\n" << fiber_array.at(j_fib) << "\n";
                throw std::runtime_error("Multiple fibers bound to site.\n");
            // clang-format on
        }

        occupied[site_pair] = i_fib;
    }

    // Pass through fibers for unassigned bodies
    std::vector<int> test_site(n_bodies); //< current site to try for insertion
    for (size_t i_fib = 0; i_fib < fiber_array.size(); ++i_fib) {
        toml::value &fiber_table = fiber_array.at(i_fib);
        int i_body = toml::find_or<int>(fiber_table, "parent_body", -1);
        int i_site = toml::find_or<int>(fiber_table, "parent_site", -1);

        // unattached or site already assigned. move to next fiber
        if (i_body < 0 || i_site >= 0) {
            Eigen::Vector3d origin{0.0, 0.0, 0.0};
            if (i_body < 0) // Initialize free fiber
                resolve_fiber_position(fiber_table, origin);
            continue;
        }

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
        toml::value &fiber_table = fiber_array.at(i_fib);
        toml::array &body_position = body_array.at(i_body).at("position").as_array();

        fiber_table["parent_body"] = i_body;
        fiber_table["parent_site"] = i_site;
        Eigen::Vector3d origin = parse_util::convert_array<>(body_position);

        resolve_fiber_position(fiber_table, origin);

        if (fiber_table.contains("x")) {
            Eigen::VectorXd x_array = parse_util::convert_array<>(fiber_table.at("x").as_array());
            nucleation_sites[i_body][i_site] = x_array.segment(0, 3) - origin;
        }
    }

    for (int i_body = 0; i_body < n_bodies; ++i_body) {
        toml::value &body_table = body_array.at(i_body);
        if (!body_table.contains("nucleation_sites"))
            body_table["nucleation_sites"] = toml::array();

        int n_nucleation_sites = toml::find_or<int>(body_table, "n_nucleation_sites", nucleation_sites[i_body].size());
        double radius = body_table["radius"].as_floating();
        // FIXME: add min_separation parameter
        const double min_separation2 = pow(params_.dynamic_instability.min_separation, 2);

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

        toml::array &x = body_table["nucleation_sites"].as_array();
        for (auto &[site, xvec] : nucleation_sites[i_body])
            for (int i = 0; i < 3; ++i)
                x.push_back(xvec[i]);
    }

    for (const auto &site_map : occupied) {
        auto [i_body, i_site] = site_map.first;
        auto i_fib = site_map.second;
        toml::value &fiber_table = fiber_array.at(i_fib);
        toml::array &body_position = body_array.at(i_body).at("position").as_array();
        toml::array &nucleation_sites = body_array.at(i_body).at("nucleation_sites").as_array();

        fiber_table["parent_body"] = i_body;
        fiber_table["parent_site"] = i_site;
        Eigen::Vector3d origin = parse_util::convert_array<>(body_position);
        Eigen::VectorXd sites = parse_util::convert_array<>(nucleation_sites);

        if (!fiber_table.contains("x")) {
            Eigen::Vector3d site_pos = Eigen::Map<Eigen::Vector3d>(sites.data() + i_site * 3, 3);
            Eigen::Vector3d fiber_origin = site_pos;
            Eigen::Vector3d fiber_orientation = fiber_origin.normalized();
            fiber_table["relative_position"] = {fiber_origin[0], fiber_origin[1], fiber_origin[2]};
            fiber_table["orientation"] = {fiber_orientation[0], fiber_orientation[1], fiber_orientation[2]};

            resolve_fiber_position(fiber_table, origin);
        }
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
void preprocess(toml::value &config) {
    spdlog::info("Preprocessing config file");

    // Body-fiber interactions through nucleation sites
    if (config.contains("fibers") && config.contains("bodies"))
        resolve_nucleation_sites(config["fibers"].as_array(), config["bodies"].as_array());
    else if (config.contains("fibers")) {
        toml::array &fiber_array = config["fibers"].as_array();
        for (size_t i_fib = 0; i_fib < fiber_array.size(); ++i_fib) {
            toml::value &fiber_table = fiber_array.at(i_fib);
            Eigen::Vector3d origin{0.0, 0.0, 0.0};
            resolve_fiber_position(fiber_table, origin);
        }
    }
}

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

        velocities_on_fiber.block(0, i_fib, 3, 1) = v_fiber;
        velocities_on_fiber(3, i_fib) = tension_condition;
        velocities_on_fiber.block(4, i_fib, 3, 1) = w_fiber;

        i_fib++;
        xt_offset += 4 * n_pts;
    }

    return velocities_on_fiber;
}

/// @brief Get number of physical nodes local to MPI rank for each object type [fibers, shell, bodies]
std::tuple<int, int, int> get_local_node_counts() {
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

/// @brief Nucleate/grow/destroy Fibers based on dynamic instability rules. See white paper for details
///
/// Modifies:
/// - FiberContainer::fibers [for nucleation/catastrophe]
/// - Fiber::v_growth_
/// - Fiber::length_
/// - Fiber::length_prev_
void dynamic_instability() {
    const double dt = properties.dt;
    FiberContainer &fc = fc_;
    BodyContainer &bc = bc_;
    Params &params = params_;
    if (params_.dynamic_instability.n_nodes == 0)
        return;

    size_t n_sites = 0;
    std::vector<int> body_offsets(bc.bodies.size() + 1);
    int i_body = 0;
    for (const auto &body : bc.bodies) {
        n_sites += body->nucleation_sites_.cols();
        body_offsets[i_body + 1] = body_offsets[i_body] + body->nucleation_sites_.cols();
        i_body++;
    }

    auto site_index = [&body_offsets](std::pair<int, int> binding_site) -> int {
        assert(binding_site.first >= 0 && binding_site.second >= 0);
        return body_offsets[binding_site.first] + binding_site.second;
    };

    auto binding_site_from_index = [&body_offsets](int index) -> std::pair<int, int> {
        for (size_t i_body = 0; i_body < body_offsets.size() - 1; ++i_body) {
            if (index < body_offsets[i_body + 1]) {
                return {i_body, index - body_offsets[i_body]};
            }
        }
        return {-1, -1};
    };

    std::vector<uint8_t> occupied_flat(bc.get_global_site_count(), 0);
    auto fib = fc.fibers.begin();
    while (fib != fc.fibers.end()) {
        fib->v_growth_ = params.dynamic_instability.v_growth;
        double f_cat = params.dynamic_instability.f_catastrophe;
        if (fib->near_periphery) {
            fib->v_growth_ *= params.dynamic_instability.v_grow_collision_scale;
            f_cat *= params.dynamic_instability.f_catastrophe_collision_scale;
        }

        if (RNG::uniform() > exp(-dt * f_cat)) {
            fib = fc.fibers.erase(fib);
        } else {
            if (fib->attached_to_body())
                occupied_flat[site_index(fib->binding_site_)] = 1;
            fib->length_prev_ = fib->length_;
            fib->length_ += dt * fib->v_growth_;
            fib++;
        }
    }

    MPI_Reduce(rank_ == 0 ? MPI_IN_PLACE : occupied_flat.data(), occupied_flat.data(), occupied_flat.size(), MPI_BYTE,
               MPI_LOR, 0, MPI_COMM_WORLD);

    std::unordered_map<int, bool> active_sites;
    std::unordered_map<int, bool> inactive_sites;
    std::vector<std::pair<int, int>> to_nucleate;
    if (rank_ == 0) {
        for (size_t i = 0; i < occupied_flat.size(); ++i) {
            if (occupied_flat[i])
                active_sites[i] = true;
            else
                inactive_sites[i] = true;
        }

        // FIXME: Is this right? I feel like the nucleation rate should be proportional to the
        // sites, or the area, or something rather than a global parameter
        int n_to_nucleate = std::min(RNG::poisson_int(dt * params.dynamic_instability.nucleation_rate),
                                     static_cast<int>(inactive_sites.size()));
        int n_trials = 100;
        while (n_to_nucleate && n_trials) {
            int passive_site_index =
                std::next(inactive_sites.begin(), RNG::uniform_int(0, inactive_sites.size()))->first;

            auto [i_body, i_site] = binding_site_from_index(passive_site_index);
            Eigen::Vector3d site_pos_i = bc.at(i_body).nucleation_sites_.col(i_site);
            bool valid_site = true;
            const double min_ds2 = pow(params.dynamic_instability.min_separation, 2);
            for (auto &[active_site_index, dum] : active_sites) {
                auto [j_body, j_site] = binding_site_from_index(active_site_index);
                Eigen::Vector3d site_pos_j = bc.at(j_body).nucleation_sites_.col(j_site);

                if ((site_pos_j - site_pos_i).squaredNorm() < min_ds2) {
                    valid_site = false;
                    break;
                }
            }

            if (valid_site) {
                n_trials = 100;
                active_sites[passive_site_index] = true;
                inactive_sites.erase(passive_site_index);
                to_nucleate.push_back({i_body, i_site});
                n_to_nucleate--;
            } else {
                n_trials--;
            }
        }
    }

    int n_fibers = fc.fibers.size();
    std::vector<int> fiber_counts(rank_ == 0 ? size_ : 0);
    MPI_Gather(&n_fibers, 1, MPI_INT, fiber_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    using fiber_struct = struct {
        int rank;
        std::pair<int, int> binding_site;
    };

    std::vector<fiber_struct> new_fibers;
    for (const auto &binding_site : to_nucleate) {
        const int fiber_rank = std::min_element(fiber_counts.begin(), fiber_counts.end()) - fiber_counts.begin();
        new_fibers.push_back({fiber_rank, binding_site});
        spdlog::info("Queueing fiber insertion to rank {} to site [{}, {}]", fiber_rank, binding_site.first,
                     binding_site.second);
        fiber_counts[fiber_rank]++;
    }

    int n_new = new_fibers.size();
    MPI_Bcast(&n_new, 1, MPI_INT, 0, MPI_COMM_WORLD);
    new_fibers.resize(n_new);
    MPI_Bcast(new_fibers.data(), sizeof(fiber_struct) * new_fibers.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
    if (n_new)
        spdlog::info("Sent {} fibers to nucleate", new_fibers.size());

    for (const auto &min_fib : new_fibers) {
        if (min_fib.rank == rank_) {
            Fiber fib(params.dynamic_instability.n_nodes, params.dynamic_instability.bending_rigidity, params.eta);
            fib.length_ = params.dynamic_instability.min_length;
            fib.length_prev_ = params.dynamic_instability.min_length;
            fib.v_growth_ = 0.0;
            fib.binding_site_ = min_fib.binding_site;

            Eigen::MatrixXd x(3, fib.n_nodes_);
            Eigen::ArrayXd s = Eigen::ArrayXd::LinSpaced(fib.n_nodes_, 0, fib.length_).transpose();
            Eigen::Vector3d origin = bc_.get_nucleation_site(fib.binding_site_.first, fib.binding_site_.second);
            Eigen::Vector3d u = (origin - bc_.bodies[fib.binding_site_.first]->get_position()).normalized();

            for (int i = 0; i < 3; ++i)
                fib.x_.row(i) = origin(i) + u(i) * s;

            fc.fibers.push_back(fib);
            spdlog::get("SkellySim global")
                ->info("Inserted fiber on rank {} at site [{}, {}]", rank_, min_fib.binding_site.first,
                       min_fib.binding_site.second);
        }
    }
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
    Block<MatrixXd> r_shellbody = r_all.block(0, r_fibers.cols(), 3, r_shell.cols() + r_bodies.cols());
    MatrixXd v_fib2all = fc.flow(fw, r_shellbody, eta);

    MatrixXd r_fibbody(3, r_fibers.cols() + r_bodies.cols());
    r_fibbody.block(0, 0, 3, r_fibers.cols()) = r_fibers;
    r_fibbody.block(0, r_fibers.cols(), 3, r_bodies.cols()) = r_bodies;
    MatrixXd v_shell2fibbody = shell.flow(r_fibbody, x_shell, eta);

    MatrixXd v_fib_boundary = System::calculate_body_fiber_link_conditions(x_fibers, x_bodies);

    v_all = v_fib2all;
    v_fibers += v_shell2fibbody.block(0, 0, 3, r_fibers.cols());
    v_bodies += v_shell2fibbody.block(0, r_fibers.cols(), 3, r_bodies.cols());
    v_all += bc.flow(r_all, x_bodies, eta);

    res_fibers = fc.matvec(x_fibers, v_fibers, v_fib_boundary);
    res_shell = shell.matvec(x_shell, v_shell);
    res_bodies = bc.matvec(v_bodies, x_bodies);

    return res;
}

/// @brief Generate next trial system state for the current System::properties::dt
///
/// @note Modifies anything that evolves in time.
/// @return If the Matrix solver converged to the requested tolerance with no issue.
bool step() {
    using Eigen::MatrixXd;
    Params &params = params_;
    Periphery &shell = *shell_;
    FiberContainer &fc = fc_;
    BodyContainer &bc = bc_;
    const double eta = params.eta;
    const double dt = properties.dt;

    // Since DI can change size of fiber containers, must call first.
    System::dynamic_instability();

    const auto [fib_node_count, shell_node_count, body_node_count] = get_local_node_counts();

    MatrixXd r_trg_external(3, shell_node_count + body_node_count);
    r_trg_external.block(0, 0, 3, shell_node_count) = shell.get_local_node_positions();
    r_trg_external.block(0, shell_node_count, 3, body_node_count) = bc.get_local_node_positions(bc.bodies);

    fc.update_cache_variables(dt, eta);

    MatrixXd f_on_fibers = fc.generate_constant_force();
    MatrixXd v_all = fc.flow(f_on_fibers, r_trg_external, eta);

    bc.update_cache_variables(eta);

    // Check for an add external body forces
    bool external_force_body = false;
    for (auto &body : bc.spherical_bodies) {
        body->force_torque_.setZero();
        body->force_torque_.segment(0, 3) = body->external_force_;
        external_force_body = external_force_body | body->external_force_.any();
    }

    if (external_force_body) {
        const int total_node_count = fib_node_count + shell_node_count + body_node_count;
        MatrixXd r_all(3, total_node_count);
        auto [r_fibers, r_shell, r_bodies] = get_node_maps(r_all);
        r_fibers = fc.get_local_node_positions();
        r_shell = shell.get_local_node_positions();
        r_bodies = bc.get_local_node_positions(bc.bodies);

        v_all += bc.flow(r_all, Eigen::MatrixXd::Zero(r_bodies.rows(), r_bodies.cols()), eta);
    }

    bc.update_RHS(v_all.block(0, fib_node_count + shell_node_count, 3, body_node_count));

    fc.update_RHS(dt, v_all.block(0, 0, 3, fib_node_count), f_on_fibers);
    fc.update_boundary_conditions(shell, params.periphery_binding_flag);
    fc.apply_bc_rectangular(dt, v_all.block(0, 0, 3, fib_node_count), f_on_fibers);

    shell.update_RHS(v_all.block(0, fib_node_count, 3, shell_node_count));

    Solver<P_inv_hydro, A_fiber_hydro> solver_;
    solver_.set_RHS();
    bool converged = solver_.solve();
    CVectorMap sol = solver_.get_solution();

    double residual = solver_.get_residual();
    spdlog::info("Residual: {}", residual);

    auto [fiber_sol, shell_sol, body_sol] = get_solution_maps(sol.data());

    size_t offset = 0;
    for (auto &fib : fc.fibers) {
        for (int i = 0; i < 3; ++i)
            fib.x_.row(i) = sol.segment(offset + i * fib.n_nodes_, fib.n_nodes_);
        offset += 4 * fib.n_nodes_;
    }

    // Re-pin fibers to bodies
    for (auto &fib : fc.fibers) {
        if (fib.binding_site_.first >= 0) {
            Eigen::Vector3d delta =
                bc.get_nucleation_site(fib.binding_site_.first, fib.binding_site_.second) - fib.x_.col(0);
            fib.x_.colwise() += delta;
        }
    }

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

    System::write();
    while (properties.time < params.t_final) {
        System::backup();
        bool converged = System::step();
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

        if (accept) {
            spdlog::info("Accepting timestep and advancing time");
            properties.time += properties.dt;
            double &dt_write = params_.dt_write;
            if ((int)(properties.time / dt_write) > (int)((properties.time - properties.dt) / dt_write))
                System::write();
        } else {
            spdlog::info("Rejecting timestep");
            System::restore();
        }
        properties.dt = dt_new;
        spdlog::info("System time, dt, fiber_error: {}, {}, {}", properties.time, dt_new, fiber_error);
    }

    System::write();
}

/// @brief Check for any collisions between objects
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
void init(const std::string &input_file, bool resume_flag) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    spdlog::logger sink = rank_ == 0
                              ? spdlog::logger("SkellySim", std::make_shared<spdlog::sinks::ansicolor_stdout_sink_st>())
                              : spdlog::logger("SkellySim", std::make_shared<spdlog::sinks::null_sink_st>());
    spdlog::set_default_logger(std::make_shared<spdlog::logger>(sink));
    spdlog::stdout_color_mt("STKFMM");
    spdlog::stdout_color_mt("Belos");
    spdlog::stdout_color_mt("SkellySim global");
    spdlog::cfg::load_env_levels();

    param_table_ = toml::parse(input_file);
    params_ = Params(param_table_.at("params"));
    RNG::init(params_.seed);
    preprocess(param_table_);

    if (param_table_.contains("fibers"))
        fc_ = FiberContainer(param_table_.at("fibers").as_array(), params_);

    if (param_table_.contains("periphery")) {
        const toml::value &periphery_table = param_table_.at("periphery");
        if (!params_.shell_precompute_file.length())
            throw std::runtime_error("Periphery specified, but no precompute file. Set [params] shell_precompute_file "
                                     "in your input config and run the precompute script.");
        if (toml::find_or(periphery_table, "shape", "") == std::string("sphere"))
            shell_ = std::make_unique<SphericalPeriphery>(params_.shell_precompute_file, periphery_table, params_);
        else {
            throw std::runtime_error("Unknown shape for periphery set in input file. Valid values are 'sphere'");
        }
    } else {
        shell_ = std::make_unique<Periphery>();
    }

    if (param_table_.contains("bodies"))
        bc_ = BodyContainer(param_table_.at("bodies").as_array(), params_);
    properties.dt = params_.dt_initial;

    std::string filename = "skelly_sim.out." + std::to_string(rank_);
    if (resume_flag) {
        resume_from_trajectory(filename);
        ofs_ = std::ofstream(filename, std::ofstream::out | std::ofstream::binary | std::ofstream::app);
    } else {
        ofs_ = std::ofstream(filename, std::ofstream::out | std::ofstream::binary);
    }
}
} // namespace System
