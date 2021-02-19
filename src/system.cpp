#include <Eigen/Core>
#include <fstream>
#include <random>

#include <parse_util.hpp>
#include <system.hpp>

// TODO: Refactor all preprocess stuff. It's awful

/// RNG for generating nucleation site positions
std::mt19937 rng;
std::uniform_real_distribution<double> uniform_rng(0.0, 1.0);

/// @brief Convert fiber initial positions/orientations to full coordinate representation
/// @param[out] fiber_array Explicitly instantiated fiber array config
void resolve_fiber_positions(toml::array *fiber_array) {
    for (int i_fib = 0; i_fib < fiber_array->size(); ++i_fib) {
        toml::table *fiber_table = fiber_array->get_as<toml::table>(i_fib);
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
            Eigen::Vector3d origin = parse_util::convert_array<>(x_0);
            Eigen::Vector3d orientation = parse_util::convert_array<>(u);
            for (int i = 0; i < 3; ++i)
                x.row(i) = origin(i) + orientation(i) * Eigen::ArrayXd::LinSpaced(n_nodes, 0, length).transpose();
            for (int i = 0; i < x.size(); ++i) {
                x_array->push_back(x.data()[i]);
            }
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
    for (int i_fib = 0; i_fib < fiber_array->size(); ++i_fib) {
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
    for (int i_fib = 0; i_fib < fiber_array->size(); ++i_fib) {
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
        toml::array *x_array = fiber_table->get_as<toml::array>("x");

        fiber_table->insert_or_assign("parent_body", i_body);
        fiber_table->insert_or_assign("parent_site", i_site);

        nucleation_sites[i_body][i_site] = {x_array->get_as<double>(0)->get(), x_array->get_as<double>(1)->get(),
                                            x_array->get_as<double>(2)->get()};
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

    // Fiber positions
    if (!!fiber_array)
        resolve_fiber_positions(fiber_array);

    // Body-fiber interactions through nucleation sites
    if (!!body_array && !!fiber_array)
        resolve_nucleation_sites(fiber_array, body_array);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
System::calculate_body_fiber_link_conditions(const FiberContainer &fc, const BodyContainer &bc,
                                             const Eigen::Ref<const Eigen::VectorXd> &fibers_xt,
                                             const Eigen::Ref<const Eigen::MatrixXd> &body_velocities) {
    using Eigen::ArrayXXd;
    using Eigen::MatrixXd;
    using Eigen::Vector3d;

    Eigen::MatrixXd force_torque_on_body = MatrixXd::Zero(6, bc.get_global_count());
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
        force_torque_on_body.col(i_body).segment(0, 3) += F_body;
        force_torque_on_body.col(i_body).segment(3, 3) += L_body;

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

    return std::make_pair(force_torque_on_body, velocities_on_fiber);
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
