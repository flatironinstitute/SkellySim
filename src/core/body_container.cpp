#include <skelly_sim.hpp>

#include <body.hpp>
#include <body_container.hpp>
#include <body_deformable.hpp>
#include <body_ellipsoidal.hpp>
#include <body_spherical.hpp>
#include <fiber_container_finite_difference.hpp>
#include <kernels.hpp>
#include <system.hpp>
#include <tuple>
#include <utils.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/// @brief Update the RHS for all bodies for given velocities
///
/// @param[in] v_on_bodies [3 x n_local_body_nodes] matrix of velocities at the body nodes
void BodyContainer::update_RHS(MatrixRef &v_on_bodies) {
    if (world_rank_)
        return;
    int offset = 0;
    for (auto &body : bodies) {
        body->update_RHS(v_on_bodies.block(0, offset, 3, body->n_nodes_));
        offset += body->n_nodes_;
    }
}

/// @brief Return a copy of the current RHS for all bodies (all bodies on rank 0, empty otherwise)
///
/// @return [local_solution_size] vector of the RHS for all bodies
VectorXd BodyContainer::get_RHS() const {
    VectorXd RHS(get_local_solution_size());

    if (world_rank_ == 0) {
        int offset = 0;
        for (const auto &body : bodies) {
            RHS.segment(offset, body->RHS_.size()) = body->RHS_;
            offset += body->RHS_.size();
        }
    }
    return RHS;
}

void BodyContainer::step(VectorRef &bodies_solution, double dt) const {
    const int global_solution_size = get_global_solution_size();
    VectorXd bodies_solution_global(global_solution_size);
    if (world_rank_ == 0)
        bodies_solution_global = bodies_solution;
    MPI_Bcast(bodies_solution_global.data(), global_solution_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int sol_offset = 0;
    for (auto &body : bodies) {
        const int sol_size = body->get_solution_size();
        body->step(dt, bodies_solution_global.segment(sol_offset, sol_size));
        sol_offset += sol_size;
    }
}

/// @brief Return copy of local node positions (all bodies on rank 0, empty otherwise)
/// @param[in] std::vector of shared_ptr<DerivedBody>
/// @return [3 x n_body_nodes_local] matrix of body node positions
template <typename T>
MatrixXd BodyContainer::get_local_node_positions(const T &body_vec) const {
    const int n_nodes = get_local_node_count();
    MatrixXd r_body_nodes(3, n_nodes);

    if (world_rank_ == 0) {
        int offset = 0;
        for (const auto &body : body_vec) {
            r_body_nodes.block(0, offset, 3, body->n_nodes_) = body->node_positions_;
            offset += body->n_nodes_;
        }
    }

    return r_body_nodes;
}
template MatrixXd BodyContainer::get_local_node_positions(const decltype(BodyContainer::bodies) &) const;

/// @brief Return copy of local node positions (all bodies on rank 0, empty otherwise)
/// @param[in] std::vector of shared_ptr<DerivedBody>
/// @return [3 x n_body_nodes_local] matrix of body node positions
template <typename T>
MatrixXd BodyContainer::get_local_node_normals(const T &body_vec) const {
    const int n_nodes = get_local_node_count();
    MatrixXd normals(3, n_nodes);

    if (world_rank_ == 0) {
        int offset = 0;
        for (const auto &body : body_vec) {
            normals.block(0, offset, 3, body->n_nodes_) = body->node_normals_;
            offset += body->n_nodes_;
        }
    }

    return normals;
}

template <typename T>
VectorXd BodyContainer::get_local_solution(const T &body_vec, VectorRef &body_solutions) const {
    VectorXd sub_solution;

    if (world_rank_ == 0) {
        int solution_size = 0;
        for (const auto &body : body_vec)
            solution_size += body->get_solution_size();
        sub_solution.resize(solution_size);

        int solution_offset = 0;
        for (auto &body : body_vec) {
            const int body_solution_size = body->get_solution_size();
            sub_solution.segment(solution_offset, body_solution_size) =
                body_solutions.segment(solution_offsets_.at(body), body_solution_size);
            solution_offset += body->get_solution_size();
        }
    }

    return sub_solution;
}

/// @brief Get forces and torques from each rank
/// @param[in] [6 x n_bodies] matrix of forces/torques LOCAL TO EACH RANK (result is summed)
/// @return pair<[3 x n_bodies_local], [3 x n_bodies_local]> matrices of body forces/torques
template <typename T>
std::pair<MatrixXd, MatrixXd> BodyContainer::get_global_forces_torques(const T &link_conditions) const {
    MatrixXd forces_torques = link_conditions;
    MPI_Allreduce(MPI_IN_PLACE, forces_torques.data(), forces_torques.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return std::make_pair(forces_torques.block(0, 0, 3, bodies.size()), forces_torques.block(3, 0, 3, bodies.size()));
}

/// @brief Get center positions regardless of rank
/// @param[in] std::vector of shared_ptr<DerivedBody>
/// @return [3 x n_bodies_local] matrix of body centers
template <typename T>
MatrixXd BodyContainer::get_global_center_positions(const T &body_vec) const {
    const int n_bodies = body_vec.size();
    Eigen::MatrixXd centers(3, n_bodies);
    for (size_t i = 0; i < body_vec.size(); ++i)
        centers.col(i) = body_vec[i]->get_position();

    return centers;
}

/// @brief Get center positions on rank 0, otherwise empty
/// @param[in] std::vector of shared_ptr<DerivedBody>
/// @return [3 x n_bodies_local] matrix of body centers
template <typename T>
MatrixXd BodyContainer::get_local_center_positions(const T &body_vec) const {
    if (world_rank_ != 0)
        return Eigen::MatrixXd();

    const int n_bodies = body_vec.size();
    Eigen::MatrixXd centers(3, n_bodies);
    for (size_t i = 0; i < body_vec.size(); ++i)
        centers.col(i) = body_vec[i]->get_position();

    return centers;
}

/// @brief Calculate forces/torques on the bodies and velocities on the fibers due to attachment constraints
/// @param[in] fibers_xt [4 x num_fiber_nodes_local] Vector of fiber node positions and tensions on current rank.
/// Ordering is [fib1.nodes.x, fib1.nodes.y, fib1.nodes.z, fib1.T, fib2.nodes.x, ...]
/// @param[in] x_bodies entire body component of the solution vector (deformable+rigid)
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
BodyContainer::calculate_link_conditions(VectorRef &fiber_sol, VectorRef &x_bodies,
                                         const FiberContainerFiniteDifference &fc) const {
    using Eigen::ArrayXXd;
    using Eigen::MatrixXd;
    using Eigen::Vector3d;

    MatrixXd velocities_on_fiber = MatrixXd::Zero(7, fc.get_local_fiber_count());
    MatrixXd body_forces_torques =
        MatrixXd::Zero(6, spherical_bodies.size() + deformable_bodies.size() + ellipsoidal_bodies.size());

    MatrixXd body_velocities(6, spherical_bodies.size() + deformable_bodies.size() + ellipsoidal_bodies.size());
    int index = 0;
    // FIXME XXX Again, this will bite us, as duplicated code (and deformable bodies shouldn't be here anyway for now)
    for (const auto &body : spherical_bodies) {
        body_velocities.col(index) =
            x_bodies.segment(solution_offsets_.at(std::static_pointer_cast<Body>(body)) + body->n_nodes_ * 3, 6);
        index++;
    }
    index += deformable_bodies.size();
    for (const auto &body : ellipsoidal_bodies) {
        body_velocities.col(index) =
            x_bodies.segment(solution_offsets_.at(std::static_pointer_cast<Body>(body)) + body->n_nodes_ * 3, 6);
        index++;
    }

    int xt_offset = 0;
    int i_fib = 0;
    for (const auto &fib : fc.fibers_) {
        const auto &fib_mats = fib.matrices_.at(fib.n_nodes_);
        const int n_pts = fib.n_nodes_;

        auto &[i_body, i_site] = fib.binding_site_;
        if (i_body < 0)
            continue;

        // FIXME XXX Eventually this will come to bite us, as the ellipsoidal bodies also could have fibers they just
        // don't for now.
        auto body = std::static_pointer_cast<SphericalBody>(bodies[i_body]);
        Vector3d site_pos = get_nucleation_site(i_body, i_site) - body->get_position();
        MatrixXd x_new(3, fib.n_nodes_);
        x_new.row(0) = fiber_sol.segment(xt_offset + 0 * fib.n_nodes_, fib.n_nodes_);
        x_new.row(1) = fiber_sol.segment(xt_offset + 1 * fib.n_nodes_, fib.n_nodes_);
        x_new.row(2) = fiber_sol.segment(xt_offset + 2 * fib.n_nodes_, fib.n_nodes_);

        double T_new_0 = fiber_sol(xt_offset + 3 * n_pts);

        Vector3d xs_0 = fib.xs_.col(0);
        MatrixXd xss_new = pow(2.0 / fib.length_, 2) * x_new * fib_mats.D_2_0;
        MatrixXd xsss_new = pow(2.0 / fib.length_, 3) * x_new * fib_mats.D_3_0;
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
        body_forces_torques.col(i_body).segment(0, 3) += F_body;
        body_forces_torques.col(i_body).segment(3, 3) += L_body;

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

    return std::make_tuple(std::move(velocities_on_fiber), std::move(body_forces_torques));
}

MatrixXd BodyContainer::flow_spherical(MatrixRef &r_trg, VectorRef &body_solutions, MatrixRef &link_conditions,
                                       double eta) const {
    spdlog::debug("Started body (spherical) flow");
    utils::LoggerRedirect redirect(std::cout);
    if (!spherical_bodies.size()) {
        spdlog::debug("Finished body (spherical) flow (no spherical bodies)");
        return MatrixXd::Zero(3, r_trg.cols());
    }

    const VectorXd spherical_solution = get_local_solution(spherical_bodies, body_solutions);
    const MatrixXd node_positions = get_local_node_positions(spherical_bodies);
    const MatrixXd node_normals = get_local_node_normals(spherical_bodies);
    const int n_nodes = node_positions.cols();
    MatrixXd densities(3, n_nodes);
    int node_offset = 0;
    if (world_rank_ == 0) {
        for (auto &body : spherical_bodies) {
            densities.block(0, node_offset, 3, body->n_nodes_) =
                CMatrixMap(body_solutions.data() + solution_offsets_.at(body), 3, body->n_nodes_);
            node_offset += body->n_nodes_;
        }
    }
    const MatrixXd null_matrix; //< Empty matrix for dummy arguments to kernels

    // Section: Stresslet kernel
    const MatrixXd &r_dl = node_positions; //< "double layer" positions for stresslet kernel
    MatrixXd f_dl(9, n_nodes);             //< "double layer" "force" for stresslet kernel

    // double layer density is 2 * outer product of normals with density
    for (int node = 0; node < n_nodes; ++node)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                f_dl(i * 3 + j, node) = 2.0 * node_normals(i, node) * densities(j, node) * eta;

    spdlog::debug("  body_stresslet");
    MatrixXd v_bdy2all = stresslet_kernel_(null_matrix, r_dl, r_trg, null_matrix, f_dl, eta);
    redirect.flush(spdlog::level::debug, "STKFMM");

    // Section: Oseen kernel
    spdlog::debug("  body_oseen");
    MatrixXd center_positions =
        get_local_center_positions(spherical_bodies); //< Distributed center positions for FMM calls

    auto [forces, torques] = get_global_forces_torques(link_conditions);

    // We actually only need the summed forces on the first rank
    if (world_rank_)
        forces.resize(3, 0);
    v_bdy2all += stokeslet_kernel_(center_positions, null_matrix, r_trg, forces, null_matrix, eta);
    redirect.flush(spdlog::level::debug, "STKFMM");

    // Since rotlet isn't handled via an FMM we don't distribute the nodes, but instead each
    // rank gets the body centers and calculates the center->target rotlet
    spdlog::debug("  body_rotlet");
    center_positions = get_global_center_positions(spherical_bodies);

    v_bdy2all += kernels::rotlet(center_positions, r_trg, torques, eta);

    spdlog::debug("Finished body (spherical) flow");
    return v_bdy2all;
}

MatrixXd BodyContainer::flow_ellipsoidal(MatrixRef &r_trg, VectorRef &body_solutions, MatrixRef &link_conditions,
                                         double eta) const {
    spdlog::debug("Started body (ellipsoidal) flow");
    utils::LoggerRedirect redirect(std::cout);
    if (!ellipsoidal_bodies.size()) {
        spdlog::debug("Finished body (ellipsoidal) flow (no ellipsoidal bodies)");
        return MatrixXd::Zero(3, r_trg.cols());
    }

    const VectorXd ellipsoidal_solution = get_local_solution(ellipsoidal_bodies, body_solutions);
    const MatrixXd node_positions = get_local_node_positions(ellipsoidal_bodies);
    const MatrixXd node_normals = get_local_node_normals(ellipsoidal_bodies);
    const int n_nodes = node_positions.cols();
    MatrixXd densities(3, n_nodes);
    int node_offset = 0;
    if (world_rank_ == 0) {
        for (auto &body : ellipsoidal_bodies) {
            densities.block(0, node_offset, 3, body->n_nodes_) =
                CMatrixMap(body_solutions.data() + solution_offsets_.at(body), 3, body->n_nodes_);
            node_offset += body->n_nodes_;
        }
    }
    const MatrixXd null_matrix; //< Empty matrix for dummy arguments to kernels

    // Section: Stresslet kernel
    const MatrixXd &r_dl = node_positions; //< "double layer" positions for stresslet kernel
    MatrixXd f_dl(9, n_nodes);             //< "double layer" "force" for stresslet kernel

    // double layer density is 2 * outer product of normals with density
    for (int node = 0; node < n_nodes; ++node)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                f_dl(i * 3 + j, node) = 2.0 * node_normals(i, node) * densities(j, node) * eta;

    spdlog::debug("  body_stresslet");
    MatrixXd v_bdy2all = stresslet_kernel_(null_matrix, r_dl, r_trg, null_matrix, f_dl, eta);
    redirect.flush(spdlog::level::debug, "STKFMM");

    // Section: Oseen kernel
    spdlog::debug("  body_oseen");
    MatrixXd center_positions =
        get_local_center_positions(ellipsoidal_bodies); //< Distributed center positions for FMM calls

    auto [forces, torques] = get_global_forces_torques(link_conditions);

    // We actually only need the summed forces on the first rank
    if (world_rank_)
        forces.resize(3, 0);
    v_bdy2all += stokeslet_kernel_(center_positions, null_matrix, r_trg, forces, null_matrix, eta);
    redirect.flush(spdlog::level::debug, "STKFMM");

    // Since rotlet isn't handled via an FMM we don't distribute the nodes, but instead each
    // rank gets the body centers and calculates the center->target rotlet
    spdlog::debug("  body_rotlet");
    center_positions = get_global_center_positions(ellipsoidal_bodies);

    v_bdy2all += kernels::rotlet(center_positions, r_trg, torques, eta);

    spdlog::debug("Finished body (ellipsoidal) flow");
    return v_bdy2all;
}

Eigen::MatrixXd BodyContainer::calculate_external_forces_torques(double time) const {
    // Total body sizes that we know about
    MatrixXd forces_torques =
        MatrixXd::Zero(6, spherical_bodies.size() + deformable_bodies.size() + ellipsoidal_bodies.size());
    int i_body = 0;
    for (auto &body : spherical_bodies) {
        if (body->external_force_type_ == Body::EXTFORCE::Linear) {
            forces_torques.col(i_body).segment(0, 3) += body->external_force_;
        } else if (body->external_force_type_ == Body::EXTFORCE::Oscillatory) {
            forces_torques.col(i_body).segment(0, 3) +=
                body->extforce_oscillation_amplitude_ *
                std::sin(body->extforce_oscillation_omega_ * time - body->extforce_oscillation_phase_) *
                body->external_force_;
        }
        forces_torques.col(i_body).segment(3, 3) += body->external_torque_;
        i_body++;
    }

    // FIXME This shouldn't be duplicated, come back later and abstract
    i_body += deformable_bodies.size();
    for (auto &body : ellipsoidal_bodies) {
        if (body->external_force_type_ == Body::EXTFORCE::Linear) {
            forces_torques.col(i_body).segment(0, 3) += body->external_force_;
        } else if (body->external_force_type_ == Body::EXTFORCE::Oscillatory) {
            forces_torques.col(i_body).segment(0, 3) +=
                body->extforce_oscillation_amplitude_ *
                std::sin(body->extforce_oscillation_omega_ * time - body->extforce_oscillation_phase_) *
                body->external_force_;
        }
        forces_torques.col(i_body).segment(3, 3) += body->external_torque_;
        i_body++;
    }

    return forces_torques;
}

MatrixXd BodyContainer::flow_deformable(MatrixRef &r_trg, VectorRef &body_solutions, MatrixRef &link_conditions,
                                        double eta) const {
    if (!deformable_bodies.size())
        return MatrixXd::Zero(3, r_trg.cols());
    throw std::runtime_error("BodyContainer::flow_deformable not yet supported.");
    utils::LoggerRedirect redirect(std::cout);

    const VectorXd spherical_solution = get_local_solution(deformable_bodies, body_solutions);

    // Code here
    MatrixXd v_bdy2all = MatrixXd::Zero(3, r_trg.cols());

    redirect.flush(spdlog::level::debug, "STKFMM");
    return v_bdy2all;
}

/// @brief Calculate velocity at target coordinates due to the bodies
/// @param[in] r_trg [3 x n_trg_local] Matrix of target coordinates to evaluate the velocity due to bodies at
/// @param[in] densities [3 x n_nodes_local] Matrix of body node source strengths
/// @param[in] forces_torques [6 x n_bodies] Matrix of body center-of-mass forces and torques
/// @param[in] eta Fluid viscosity
/// @return [3 x n_trg_local] Matrix of velocities due to bodies at target coordinates
MatrixXd BodyContainer::flow(MatrixRef &r_trg, VectorRef &body_solutions, MatrixRef &link_conditions,
                             double eta) const {
    MatrixXd v_spherical = flow_spherical(r_trg, body_solutions, link_conditions, eta);
    MatrixXd v_deformable = flow_deformable(r_trg, body_solutions, link_conditions, eta);
    MatrixXd v_ellipsoidal = flow_ellipsoidal(r_trg, body_solutions, link_conditions, eta);
    return v_spherical + v_deformable + v_ellipsoidal;
}

/// @brief Apply body portion of matrix-free operator given densities/velocities
///
/// \f[ A_{\textrm{bodies}} * x_\textrm{bodies} = y_{\textrm{bodies}} \f]
/// where 'x' is derived from the input densities and velocities
/// @param[in] v_bodies [3 x n_body_nodes_local] matrix of velocities at body nodes
/// @param[in] x_bodies [3 * n_body_nodes_local + 6 * n_bodies_global] vector of body source strength "densities", then
/// com velocities, then com angular velocities
/// @return [body_local_solution_size] vector 'y' in the formulation above
VectorXd BodyContainer::matvec(MatrixRef &v_bodies, VectorRef &x_bodies) const {
    VectorXd res(get_local_solution_size());
    if (world_rank_ == 0) {
        int node_offset = 0;
        int solution_offset = 0;
        for (const auto &body : bodies) {
            const int sol_size = body->get_solution_size();
            CVectorMap x_body(x_bodies.data() + solution_offset, sol_size);
            VectorMap res_body(res.data() + solution_offset, sol_size);
            Eigen::VectorXd res_body_tmp = body->matvec(v_bodies.block(0, node_offset, 3, body->n_nodes_), x_body);
            res_body = res_body_tmp;

            solution_offset += sol_size;
            node_offset += body->n_nodes_;
        }
    }
    return res;
}

/// @brief Apply body portion of preconditioner given linearized body state vector x
///
/// \f[ P^{-1}_{\textrm{bodies}} * x_\textrm{bodies} = y_{\textrm{bodies}} \f]
/// @return [body_local_solution_size] preconditioned vector 'y' in the formulation above
VectorXd BodyContainer::apply_preconditioner(VectorRef &x) const {
    VectorXd res(get_local_solution_size());
    if (world_rank_ == 0) {
        int offset = 0;
        for (const auto &b : bodies) {
            const int blocksize = b->get_solution_size();
            res.segment(offset, blocksize) = b->apply_preconditioner(x.segment(offset, blocksize));
            offset += blocksize;
        }
    }
    return res;
}

void BodyContainer::populate_sublists() {
    using std::static_pointer_cast;
    spherical_bodies.clear();
    deformable_bodies.clear();
    ellipsoidal_bodies.clear();
    solution_offsets_.clear();
    node_offsets_.clear();
    int solution_offset = 0;
    int node_offset = 0;
    for (const auto &body : bodies) {
        if (dynamic_cast<SphericalBody *>(body.get())) {
            spherical_bodies.push_back(static_pointer_cast<SphericalBody>(body));
        } else if (dynamic_cast<DeformableBody *>(body.get())) {
            deformable_bodies.push_back(static_pointer_cast<DeformableBody>(body));
        } else if (dynamic_cast<EllipsoidalBody *>(body.get())) {
            ellipsoidal_bodies.push_back(static_pointer_cast<EllipsoidalBody>(body));
        }

        solution_offsets_[body] = solution_offset;
        solution_offset += body->get_solution_size();

        node_offsets_[body] = node_offset;
        node_offset += body->n_nodes_;
    }
}

void BodyContainer::set_evaluator(const std::string &evaluator) {
    auto &params = *System::get_params();

    if (evaluator == "FMM") {
        auto &sp = params.stkfmm;
        utils::LoggerRedirect redirect(std::cout);
        stresslet_kernel_ =
            kernels::FMM<stkfmm::Stk3DFMM>(sp.body_stresslet_multipole_order, sp.body_stresslet_max_points,
                                           stkfmm::PAXIS::NONE, stkfmm::KERNEL::PVel, kernels::stokes_pvel_fmm);
        redirect.flush(spdlog::level::debug, "STKFMM");
        stokeslet_kernel_ =
            kernels::FMM<stkfmm::Stk3DFMM>(sp.body_oseen_multipole_order, sp.body_oseen_max_points, stkfmm::PAXIS::NONE,
                                           stkfmm::KERNEL::Stokes, kernels::stokes_vel_fmm);
        redirect.flush(spdlog::level::debug, "STKFMM");
    } else if (evaluator == "CPU") {
        stresslet_kernel_ = kernels::stresslet_direct_cpu;
        stokeslet_kernel_ = kernels::stokeslet_direct_cpu;
    } else if (evaluator == "GPU") {
        stresslet_kernel_ = kernels::stresslet_direct_gpu;
        stokeslet_kernel_ = kernels::stokeslet_direct_gpu;
    }
}

// FIXME: remove redundant code in =/copy
/// @brief Copy constructor...
BodyContainer::BodyContainer(const BodyContainer &orig) {
    for (auto &body : orig.bodies) {
        bodies.push_back(body->clone());
    }
    world_rank_ = orig.world_rank_;
    world_size_ = orig.world_size_;
    stresslet_kernel_ = orig.stresslet_kernel_;
    stokeslet_kernel_ = orig.stokeslet_kernel_;
    populate_sublists();
};

/// @brief Assignment operator...
BodyContainer &BodyContainer::operator=(const BodyContainer orig) {
    bodies.clear();
    for (auto &body : orig.bodies) {
        bodies.push_back(body->clone());
    }
    world_rank_ = orig.world_rank_;
    world_size_ = orig.world_size_;
    stresslet_kernel_ = orig.stresslet_kernel_;
    stokeslet_kernel_ = orig.stokeslet_kernel_;
    populate_sublists();
    return *this;
}

/// @brief Construct and fill BodyContainer from parsed toml and system parameters
///
/// @param[in] body_tables Parsed TOML array of initial body objects
/// @param[in] params initialized Params struct of system parameters
BodyContainer::BodyContainer(toml::array &body_tables, Params &params) {
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);

    set_evaluator(params.pair_evaluator);

    const int n_bodies_tot = body_tables.size();
    spdlog::info("Reading in {} bodies", n_bodies_tot);
    for (int i_body = 0; i_body < n_bodies_tot; ++i_body) {
        toml::value &body_table = body_tables.at(i_body);
        const std::string shape = toml::find_or(body_table, "shape", "");
        if (shape == std::string("sphere"))
            bodies.emplace_back(new SphericalBody(body_table, params));
        else if (shape == std::string("deformable")) {
            bodies.emplace_back(new DeformableBody(body_table, params));
        } else if (shape == std::string("ellipsoid")) {
            bodies.emplace_back(new EllipsoidalBody(body_table, params));
        } else {
            throw std::runtime_error("Unknown body shape: " + shape);
        }

        auto &body = bodies.back();
        auto position = body->get_position();
        spdlog::info("  Created body {}: [ {}, {}, {} ]", i_body, position[0], position[1], position[2]);
    }

    populate_sublists();
}
