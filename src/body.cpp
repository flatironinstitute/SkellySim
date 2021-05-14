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
Eigen::VectorXd BodyContainer::get_RHS() const {
    Eigen::VectorXd RHS(get_local_solution_size());

    if (world_rank_ == 0) {
        int offset = 0;
        for (const auto &body : bodies) {
            RHS.segment(offset, body->RHS_.size()) = body->RHS_;
            offset += body->RHS_.size();
        }
    }
    return RHS;
}

/// @brief Return copy of local node positions (all bodies on rank 0, empty otherwise)
///
/// @return [3 x n_body_nodes_local] matrix of body node positions
Eigen::MatrixXd BodyContainer::get_local_node_positions() const {
    Eigen::MatrixXd r_body_nodes;

    const int n_nodes = get_local_node_count();
    r_body_nodes.resize(3, n_nodes);
    if (world_rank_ == 0) {
        r_body_nodes.resize(3, n_nodes);
        int offset = 0;
        for (const auto &body : bodies) {
            r_body_nodes.block(0, offset, 3, body->n_nodes_) = body->node_positions_;
            offset += body->n_nodes_;
        }
    }

    return r_body_nodes;
}

/// @brief Return copy of local node normals (all bodies on rank 0, empty otherwise)
///
/// @return [3 x n_body_nodes_local] matrix of body node normals
Eigen::MatrixXd BodyContainer::get_local_node_normals() const {
    Eigen::MatrixXd node_normals;

    const int n_nodes = get_local_node_count();
    node_normals.resize(3, n_nodes);
    if (world_rank_ == 0) {
        int offset = 0;
        for (const auto &body : bodies) {
            node_normals.block(0, offset, 3, body->n_nodes_) = body->node_normals_;
            offset += body->n_nodes_;
        }
    }

    return node_normals;
}

void BodyContainer::step(VectorRef &bodies_solution, double dt) const {
    const int global_solution_size = get_global_solution_size();
    Eigen::VectorXd bodies_solution_global(global_solution_size);
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

Eigen::MatrixXd BodyContainer::flow_body_spherical(const std::vector<const Body *> &bodies, MatrixRef &r_trg,
                                                   VectorRef &body_solution, double eta) const {
    spdlog::debug("Started body flow");
    utils::LoggerRedirect redirect(std::cout);
    if (!bodies.size())
        return Eigen::MatrixXd::Zero(3, r_trg.cols());
    const int n_nodes = get_local_node_count(); //< Distributed node counts for fmm calls
    const int n_trg = r_trg.cols();
    const Eigen::MatrixXd node_positions = get_local_node_positions(); //< Distributed node positions for fmm calls
    const Eigen::MatrixXd node_normals = get_local_node_normals();     //< Distributed node normals for fmm calls
    const Eigen::MatrixXd null_matrix;                                 //< Empty matrix for dummy arguments to kernels
    const int n_bodies_global = get_global_count();

    // Section: Stresslet kernel
    const Eigen::MatrixXd &r_dl = node_positions; //< "double layer" positions for stresslet kernel
    Eigen::MatrixXd f_dl(9, n_nodes);             //< "double layer" "force" for stresslet kernel

    // double layer density is 2 * outer product of normals with density
    for (int node = 0; node < n_nodes; ++node)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                f_dl(i * 3 + j, node) = 2.0 * node_normals(i, node) * densities(j, node);

    spdlog::debug("body_stresslet");
    Eigen::MatrixXd v_bdy2all =
        (*stresslet_kernel_)(null_matrix, r_dl, r_trg, null_matrix, f_dl).block(1, 0, 3, n_trg) / eta;
    redirect.flush(spdlog::level::debug, "STKFMM");

    // Section: Oseen kernel
    spdlog::debug("body_oseen");
    Eigen::MatrixXd center_positions = get_local_center_positions(); //< Distributed center positions for FMM calls
    Eigen::MatrixXd forces = forces_torques.block(0, 0, 3, center_positions.cols());
    v_bdy2all += (*oseen_kernel_)(center_positions, null_matrix, r_trg, forces, null_matrix) / eta;
    redirect.flush(spdlog::level::debug, "STKFMM");

    // Since rotlet isn't handled via an FMM we don't distribute the nodes, but instead each
    // rank gets the body centers and calculates the center->target rotlet
    spdlog::debug("body_rotlet");
    center_positions = get_global_center_positions();
    Eigen::MatrixXd torques = forces_torques.block(3, 0, 3, n_bodies_global);

    v_bdy2all += kernels::rotlet(center_positions, r_trg, torques, eta);

    spdlog::debug("Finished body flow");
    return v_bdy2all;
}

Eigen::MatrixXd BodyContainer::flow_body_deformable(const std::vector<const Body *> &bodies, MatrixRef &r_trg,
                                                    VectorRef &body_solution, double eta) const {
    return Eigen::MatrixXd::Zero();
}

/// @brief Calculate velocity at target coordinates due to the bodies
/// @param[in] r_trg [3 x n_trg_local] Matrix of target coordinates to evaluate the velocity due to bodies at
/// @param[in] densities [3 x n_nodes_local] Matrix of body node source strengths
/// @param[in] forces_torques [6 x n_bodies] Matrix of body center-of-mass forces and torques
/// @param[in] eta Fluid viscosity
/// @return [3 x n_trg_local] Matrix of velocities due to bodies at target coordinates
Eigen::MatrixXd BodyContainer::flow(MatrixRef &r_trg, VectorRef &body_solutions, double eta) const {
    std::unordered_map<std::type_index, std::vector<const Body *>> sorted_body_map;
    std::unordered_map<std::type_index, Eigen::VectorXd> solution_map;
    int solution_offset = 0;
    for (const auto &body : bodies) {
        const auto index = std::type_index(typeid(*body));
        if (sorted_body_map.find(index) == sorted_body_map.end()) {
            sorted_body_map[index] = std::vector<const Body *>();
            solution_map[index] = Eigen::VectorXd();
        }
        sorted_body_map[index].push_back(body.get());
        const int solution_size = body->get_solution_size();
        const int old_solution_size = solution_map[index].size();
        solution_map[index].resize(old_solution_size + solution_size);
        solution_map[index].segment(old_solution_size, solution_size) =
            body_solutions.segment(solution_offset, solution_size);
        solution_offset += solution_size;
    }

    for (const auto &body_index_list : sorted_body_map) {
        const auto &index = body_index_list.first;
        const auto &body_vec = body_index_list.second;
        if (index == std::type_index(typeid(SphericalBody))) {
            flow_body_spherical(body_vec, r_trg, solution_map[index], eta);
            flow_body_spherical(body_vec, r_trg, solution_map[index], eta);
        } else if (index == std::type_index(typeid(DeformableBody))) {
            flow_body_deformable(body_vec, r_trg, solution_map[index], eta);
        } else {
            throw std::runtime_error("Unknown Body type found in BodyContainer::flow routine");
        }
    }
}

/// @brief Apply body portion of matrix-free operator given densities/velocities
///
/// \f[ A_{\textrm{bodies}} * x_\textrm{bodies} = y_{\textrm{bodies}} \f]
/// where 'x' is derived from the input densities and velocities
/// @param[in] v_bodies [3 x n_body_nodes_local] matrix of velocities at body nodes
/// @param[in] body_densities [3 x n_body_nodes_local] matrix of body source strength "densities"
/// @param[in] body_velocities [6 x n_bodies_global] vector of body velocities + angular velocities,
/// @return [body_local_solution_size] vector 'y' in the formulation above
Eigen::VectorXd BodyContainer::matvec(MatrixRef &v_bodies, VectorRef &x_bodies) const {
    Eigen::VectorXd res(get_local_solution_size());
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
Eigen::VectorXd BodyContainer::apply_preconditioner(VectorRef &x) const {
    Eigen::VectorXd res(get_local_solution_size());
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
}
