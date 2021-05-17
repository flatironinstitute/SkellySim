#include <iostream>
#include <skelly_sim.hpp>

#include <body.hpp>
#include <cnpy.hpp>
#include <kernels.hpp>
#include <parse_util.hpp>
#include <periphery.hpp>
#include <utils.hpp>

#include <typeindex>

#include <spdlog/spdlog.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/// @brief Construct body from relevant toml config and system params
///
///   @param[in] body_table toml table from pre-parsed config
///   @param[in] params Pre-constructed Params object
///   surface).
///   @return Body object that has been appropriately rotated. Other internal cache variables are _not_ updated.
/// @see update_cache_variables
Body::Body(const toml::value &body_table, const Params &params) {
    using namespace parse_util;
    using std::string;
    string precompute_file = toml::find<string>(body_table, "precompute_file");
    load_precompute_data(precompute_file);
    update_cache_variables(params.eta);
}

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

    int sol_offset = get_local_solution_size();
    for (auto &body : bodies) {
        const int sol_size = body->get_solution_size();
        body->step(dt, bodies_solution.segment(sol_offset, sol_size));
        sol_offset += sol_size;
    }
}

/// @brief Return copy of local node positions (all bodies on rank 0, empty otherwise)
/// @param[in] std::vector of shared_ptr<DerivedBody>
/// @return [3 x n_body_nodes_local] matrix of body node positions
template <typename T>
MatrixXd BodyContainer::get_local_node_positions(const T &body_vec) const {
    MatrixXd r_body_nodes;
    int n_nodes = 0;
    for (auto &body : body_vec)
        n_nodes += body->n_nodes_;

    r_body_nodes.resize(3, n_nodes);
    if (world_rank_ == 0) {
        r_body_nodes.resize(3, n_nodes);
        int offset = 0;
        for (const auto &body : body_vec) {
            r_body_nodes.block(0, offset, 3, body->n_nodes_) = body->node_positions_;
            offset += body->n_nodes_;
        }
    }

    return r_body_nodes;
}

/// @brief Return copy of local node positions (all bodies on rank 0, empty otherwise)
/// @param[in] std::vector of shared_ptr<DerivedBody>
/// @return [3 x n_body_nodes_local] matrix of body node positions
template <typename T>
MatrixXd BodyContainer::get_local_node_normals(const T &body_vec) const {
    MatrixXd normals;
    int n_nodes = 0;
    for (auto &body : body_vec)
        n_nodes += body->n_nodes_;

    normals.resize(3, n_nodes);
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
    int solution_size = 0;
    int n_nodes = 0;
    for (const auto &body : body_vec)
        solution_size += body->get_solution_size();

    VectorXd sub_solution(solution_size);
    int solution_offset = 0;
    for (auto &body : body_vec) {
        const int body_solution_size = body->get_solution_size();
        sub_solution.segment(solution_offset, body_solution_size) =
            body_solutions.segment(solution_offsets_.at(body), body_solution_size);
        solution_offset += body->get_solution_size();
    }
    return sub_solution;
}

/// @brief Get forces and torques from each rank
/// @param[in] std::vector of shared_ptr<DerivedBody>
/// @return pair<[3 x n_bodies_local], [3 x n_bodies_local]> matrices of body forces/torques
template <typename T>
std::pair<MatrixXd, MatrixXd> BodyContainer::get_global_forces_torques(const T &body_vec) const {
    const int n_bodies = body_vec.size();
    Eigen::MatrixXd forces(3, n_bodies);
    Eigen::MatrixXd torques(3, n_bodies);
    Eigen::MatrixXd forces_torques(6, n_bodies);
    for (size_t i = 0; i < body_vec.size(); ++i)
        forces_torques.col(i) = body_vec[i]->force_torque_;

    MPI_Allreduce(MPI_IN_PLACE, forces_torques.data(), forces_torques.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return std::make_pair(forces_torques.block(0, 0, 3, n_bodies), forces_torques.block(3, 0, 3, n_bodies));
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

MatrixXd BodyContainer::flow_spherical(MatrixRef &r_trg, VectorRef &body_solutions, double eta) const {
    spdlog::debug("Started body flow");
    utils::LoggerRedirect redirect(std::cout);
    if (!spherical_bodies.size())
        return MatrixXd::Zero(3, r_trg.cols());

    const VectorXd spherical_solution = get_local_solution(spherical_bodies, body_solutions);
    const MatrixXd node_positions = get_local_node_positions(spherical_bodies);
    const MatrixXd node_normals = get_local_node_normals(spherical_bodies);
    const int n_nodes = node_positions.cols();
    MatrixXd densities(3, n_nodes);
    int node_offset = 0;
    for (auto &body : spherical_bodies) {
        densities.block(0, node_offset, 3, body->n_nodes_) =
            CMatrixMap(body_solutions.data() + solution_offsets_.at(body), 3, body->n_nodes_);
        node_offset += body->n_nodes_;
    }
    const int n_trg = r_trg.cols();
    const MatrixXd null_matrix; //< Empty matrix for dummy arguments to kernels

    // Section: Stresslet kernel
    const MatrixXd &r_dl = node_positions; //< "double layer" positions for stresslet kernel
    MatrixXd f_dl(9, n_nodes);             //< "double layer" "force" for stresslet kernel

    // double layer density is 2 * outer product of normals with density
    for (int node = 0; node < n_nodes; ++node)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                f_dl(i * 3 + j, node) = 2.0 * node_normals(i, node) * densities(j, node);

    spdlog::debug("body_stresslet");
    MatrixXd v_bdy2all = (*stresslet_kernel_)(null_matrix, r_dl, r_trg, null_matrix, f_dl).block(1, 0, 3, n_trg) / eta;
    redirect.flush(spdlog::level::debug, "STKFMM");

    // Section: Oseen kernel
    spdlog::debug("body_oseen");
    MatrixXd center_positions =
        get_local_center_positions(spherical_bodies); //< Distributed center positions for FMM calls
    MatrixXd forces, torques;
    std::tie(forces, torques) = get_global_forces_torques(spherical_bodies);
    v_bdy2all += (*oseen_kernel_)(center_positions, null_matrix, r_trg, forces, null_matrix) / eta;
    redirect.flush(spdlog::level::debug, "STKFMM");

    // Since rotlet isn't handled via an FMM we don't distribute the nodes, but instead each
    // rank gets the body centers and calculates the center->target rotlet
    spdlog::debug("body_rotlet");
    center_positions = get_global_center_positions(spherical_bodies);

    v_bdy2all += kernels::rotlet(center_positions, r_trg, torques, eta);

    spdlog::debug("Finished body flow");
    return v_bdy2all;
}

MatrixXd BodyContainer::flow_deformable(MatrixRef &r_trg, VectorRef &body_solutions, double eta) const {
    if (!deformable_bodies.size())
        return MatrixXd::Zero(3, r_trg.cols());
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
MatrixXd BodyContainer::flow(MatrixRef &r_trg, VectorRef &body_solutions, double eta) const {
    MatrixXd v_spherical = flow_spherical(r_trg, body_solutions, eta);
    MatrixXd v_deformable = flow_deformable(r_trg, body_solutions, eta);
    return v_spherical + v_deformable;
}

/// @brief Apply body portion of matrix-free operator given densities/velocities
///
/// \f[ A_{\textrm{bodies}} * x_\textrm{bodies} = y_{\textrm{bodies}} \f]
/// where 'x' is derived from the input densities and velocities
/// @param[in] v_bodies [3 x n_body_nodes_local] matrix of velocities at body nodes
/// @param[in] body_densities [3 x n_body_nodes_local] matrix of body source strength "densities"
/// @param[in] body_velocities [6 x n_bodies_global] vector of body velocities + angular velocities,
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
            res_body = body->matvec(v_bodies.block(0, node_offset, 3, body->n_nodes_), x_body);

            solution_offset += sol_size;
            node_offset += 3 * body->n_nodes_;
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
    solution_offsets_.clear();
    node_offsets_.clear();
    int solution_offset = 0;
    int node_offset = 0;
    for (const auto &body : bodies) {
        using std::type_index;
        if (typeid(*body) == typeid(SphericalBody)) {
            spherical_bodies.push_back(static_pointer_cast<SphericalBody>(body));
        } else if (typeid(*body) == typeid(DeformableBody)) {
            deformable_bodies.push_back(static_pointer_cast<DeformableBody>(body));
        }

        solution_offsets_[body] = solution_offset;
        solution_offset += body->get_solution_size();

        node_offsets_[body] = node_offset;
        node_offset += body->n_nodes_;
    }
}

/// @brief Construct and fill BodyContainer from parsed toml and system parameters
///
/// @param[in] body_tables Parsed TOML array of initial body objects
/// @param[in] params initialized Params struct of system parameters
BodyContainer::BodyContainer(toml::array &body_tables, Params &params) {
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);

    // TODO: Make mult_order and max_pts passable fmm parameters
    {
        utils::LoggerRedirect redirect(std::cout);
        stresslet_kernel_ = std::unique_ptr<kernels::FMM<stkfmm::Stk3DFMM>>(new kernels::FMM<stkfmm::Stk3DFMM>(
            8, 2000, stkfmm::PAXIS::NONE, stkfmm::KERNEL::PVel, kernels::stokes_pvel_fmm));
        redirect.flush(spdlog::level::debug, "STKFMM");
        oseen_kernel_ = std::unique_ptr<kernels::FMM<stkfmm::Stk3DFMM>>(new kernels::FMM<stkfmm::Stk3DFMM>(
            8, 2000, stkfmm::PAXIS::NONE, stkfmm::KERNEL::Stokes, kernels::stokes_vel_fmm));
        redirect.flush(spdlog::level::debug, "STKFMM");
    }

    const int n_bodies_tot = body_tables.size();
    spdlog::info("Reading in {} bodies", n_bodies_tot);

    for (int i_body = 0; i_body < n_bodies_tot; ++i_body) {
        toml::value &body_table = body_tables.at(i_body);
        bodies.emplace_back(new SphericalBody(body_table, params));
        // auto &body = bodies.back();
        // spdlog::info("Body {}: {} [ {}, {}, {} ]", i_body, body->node_weights_.size(), body->position_[0],
        //              body->position_[1], body->position_[2]);
    }

    populate_sublists();
}
