#include <skelly_sim.hpp>

#include <fiber_container_base.hpp>
#include <system.hpp>
#include <utils.hpp>

/// @brief Constructor
FiberContainerBase::FiberContainerBase(toml::array &fiber_tables, Params &params) {

    spdlog::debug("FiberContainerBase::FiberContainerBase");

    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);

    set_evaluator(params.pair_evaluator);

    spdlog::debug("FiberContainerBase::FiberContainerBase return");
}

/// @brief Set the evaluator
void FiberContainerBase::set_evaluator(const std::string &evaluator) {
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

/// @brief Get the global number of fibers in all MPI ranks
int FiberContainerBase::get_global_fiber_number() const {
    const int local_fib_count = get_local_fiber_number();
    int global_fib_count;

    MPI_Allreduce(&local_fib_count, &global_fib_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return global_fib_count;
}

/// @brief Get the global number of nodes for fibers in all MPI ranks
int FiberContainerBase::get_global_node_count() const {
    const int local_node_count = get_local_node_count();
    int global_node_count;

    MPI_Allreduce(&local_node_count, &global_node_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return global_node_count;
}

/// @brief Get the global solution size across fibers in all MPI ranks
int FiberContainerBase::get_global_solution_size() const {
    const int local_solution_size = get_local_solution_size();
    int global_solution_size;

    MPI_Allreduce(&local_solution_size, &global_solution_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return global_solution_size;
}

/// @brief Set the local variables for fiber number, etc
void FiberContainerBase::set_local_fiber_numbers(int n_fibers, int n_local_nodes, int n_local_solution_size) {
    n_local_fibers_ = n_fibers;
    n_local_node_count_ = n_local_nodes;
    n_local_solution_size_ = n_local_solution_size;
}
