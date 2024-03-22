/// \file performance_hydrodynamics_combined.cpp
/// \brief Testing the performance of various implementations of the hydrodynamics kernels

// C++ includes
#include <iostream>
#include <omp.h>
#include <unordered_map>

// External includes
#include <Kokkos_Core.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

// Skelly includes
#include <params.hpp>
#include <periphery.hpp>
#include <utils.hpp>

using Eigen::MatrixXd;

int main(int argc, char *argv[]) {
    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);

    // create a kokkos environment too
    {
        const char *testargv[] = {"--kokkos-disable-warnings"};
        int testargc = 1;
        Kokkos::initialize(testargc, const_cast<char **>(testargv));
    }

    // Get mpi information
    int mpirank;
    int mpisize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

    // Setup the loggers, etc
    spdlog::logger sink = mpirank == 0
                              ? spdlog::logger("SkellySim", std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>())
                              : spdlog::logger("SkellySim", std::make_shared<spdlog::sinks::null_sink_st>());
    spdlog::set_default_logger(std::make_shared<spdlog::logger>(sink));
    spdlog::stderr_color_mt("STKFMM");
    spdlog::stderr_color_mt("Belos");
    spdlog::stderr_color_mt("SkellySim global");
    spdlog::cfg::load_env_levels();

    std::string input_file;
    double omega;
    int ntrials;
    std::string evaluator;

    Teuchos::CommandLineProcessor cmdp(false, true);
    cmdp.setOption("input_file", &input_file, "configuration file", true);
    cmdp.setOption("omega", &omega, "omega_0", true);
    cmdp.setOption("ntrials", &ntrials, "ntrials", true);
    cmdp.setOption("evaluator", &evaluator, "evaluator", true);

    if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Set up facsimiles to what SkellySim does
    Params params;
    toml::value param_table;
    std::unique_ptr<Periphery> shell;

    // Here is the current, MPI-rank solution
    Eigen::VectorXd curr_solution;

    try {
        // Load in the parameters
        toml::value param_table;
        param_table = toml::parse(input_file);
        params = Params(param_table.at("params"));
        // Overwrite the evaluator so we can control via the command line
        params.pair_evaluator = evaluator;
        params.print();

        const double eta = params.eta;

        const toml::value &periphery_table = param_table.at("periphery");
        shell = std::make_unique<SphericalPeriphery>(periphery_table, params);

        // Construct a constant velocity in 'phi' on the surface of the periphery
        const double hydrodynamic_radius = static_cast<const SphericalPeriphery *>(shell.get())->radius_ * 1.04;
        spdlog::info("Hydrodynamic radius: {}", hydrodynamic_radius);

        // Get the local node count from the periphery (needed for making vectors of the correct size)
        const auto local_node_count = shell->get_local_node_count();
        spdlog::info("Periphery: n_nodes_global: {}", shell->n_nodes_global_);
        spdlog::get("SkellySim global")->info("Rank {}: local_node_count: {}", mpirank, local_node_count);

        // Set the local solution size
        const auto local_solution_size = shell->get_local_solution_size();
        curr_solution.resize(local_solution_size);
        spdlog::get("SkellySim global")->info("Rank {}: local_solution_size: {}", mpirank, local_solution_size);

        // Mirror the functionality in system::prep_state_for_solver
        Eigen::MatrixXd r_all(3, local_node_count);
        {
            auto r_shell = Eigen::Block<Eigen::MatrixXd>(r_all.derived(), 0, 0, 3, local_node_count);
            r_shell = shell->get_local_node_positions();
        }

        // Build a vector of the 'solution' that is the boundary condition for the periphery. Then send this to all of
        // the ranks
        for (auto i = 0; i < local_node_count; i++) {
            // r_all(:,i) is the XYZ position of the point
            // curr_solution(i) is the start of the solution for the point, in this case, boundary velocity
            // double theta = acos(r_all(2, i) / hydrodynamic_radius);
            double phi = atan2(r_all(1, i), r_all(0, i));
            double r = sqrt(r_all(0, i)*r_all(0, i) + r_all(1, i)*r_all(1, i));

            // We want u_phihat = R * w0 * phi_hat
            double ux = -1.0 * sin(phi) * r * omega;
            double uy = cos(phi) * r * omega;
            double uz = 0.0;

            // Load the local solution vector
            curr_solution(3 * i) = ux;
            curr_solution(3 * i + 1) = uy;
            curr_solution(3 * i + 2) = uz;
        }

        // Now, apply the operation M_inv * u_bc for the periphery
        // This looks similar to the Periphery::apply_preconditioner step
        // alias the x_local variable
        CVectorRef x_local = curr_solution;
        // This is a vector of the entire solution
        Eigen::VectorXd x_shell(3 * shell->n_nodes_global_);
        // Run a gather to get the X value across all ranks
        MPI_Allgatherv(x_local.data(), shell->node_counts_[mpirank], MPI_DOUBLE, x_shell.data(),
                       shell->node_counts_.data(), shell->node_displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        Eigen::VectorXd sigma = shell->M_inv_ * x_shell;

        // Now, look at kernel_test.cpp and find where we evaluate some kernels, do this in a way that we can time the
        // kernel execution.

        // Set up the target points
        Eigen::MatrixXd r_trg(3, 8);
        // for (int i = 0; i < 2; ++i) {
        //     for (int j = 0; j < 2; ++j) {
        //         for (int k = 0; k < 2; k++) {
        //             // Get a linear index from the points we have
        //             int idx = i + 2 * j + 2 * 2 * k;
        //             // spdlog::info("{}, {}, {} -> {}", i, j, k, idx);
        //             r_trg(0, idx) = (2 * i - 1) * (1.0 / 3.0);
        //             r_trg(1, idx) = (2 * j - 1) * (1.0 / 3.0);
        //             r_trg(2, idx) = (2 * k - 1) * (1.0 / 3.0);
        //         }
        //     }
        // }
        // Code to match what David has in Julia in order
        {
            r_trg.col(0) << 1.0, 1.0, 1.0;
            r_trg.col(1) << -1.0, 1.0, 1.0;
            r_trg.col(2) << 1.0, -1.0, 1.0;
            r_trg.col(3) << -1.0, -1.0, 1.0;
            r_trg.col(4) << 1.0, 1.0, -1.0;
            r_trg.col(5) << -1.0, 1.0, -1.0;
            r_trg.col(6) << 1.0, -1.0, -1.0;
            r_trg.col(7) << -1.0, -1.0, -1.0;

            for (int i = 0; i < 8; ++i) {
                r_trg.col(i) << r_trg.col(i) * (1.0 / 3.0);
            }
        }

        // Now we can evaluate the stresslet
        const int n_dl = sigma.size() / 3;
        // spdlog::info("n_dl: {}", n_dl);
        Eigen::MatrixXd f_dl(9, n_dl);

        // Reshape the densities for what we want
        CMatrixMap density_reshaped(sigma.data(), 3, n_dl);
        // double layer density is 2 * outer product of normals with density
        // scales with viscosity since the stresslet_kernel_ routine divides by the viscosity, and the double-layer
        // stresslet is independent of viscosity
        for (int node = 0; node < n_dl; ++node) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    f_dl(i * 3 + j, node) = 2.0 * eta * shell->node_normal_(i, node) * density_reshaped(j, node);
                }
            }
        }

        std::ostringstream trgstream;
        trgstream << r_trg;
        spdlog::info("r_trg:\n{}", trgstream.str());

        Eigen::MatrixXd r_sl, f_sl; // dummy SL positions/values

        // Now determine how we might setup the kernel calls (similar to what the periphery does with the
        // stresslet_kernel)

        // If we are an FMM, time the setup cost of the tree
        if (params.pair_evaluator == "FMM") {
            Eigen::VectorXd trial_time = Eigen::VectorXd::Zero(ntrials);

            const int mult_order = params.stkfmm.periphery_stresslet_multipole_order;
            const int max_pts = params.stkfmm.periphery_stresslet_max_points;

            for (int itrial = 0; itrial < ntrials; itrial++) {
                utils::LoggerRedirect redirect(std::cout);

                double st = omp_get_wtime();
                auto stresslet_kernel_fmm = kernels::FMM<stkfmm::Stk3DFMM>(
                    mult_order, max_pts, stkfmm::PAXIS::NONE, stkfmm::KERNEL::PVel, kernels::stokes_pvel_fmm);
                Eigen::MatrixXd inner_vel = stresslet_kernel_fmm(r_sl, shell->node_pos_, r_trg, f_sl, f_dl, eta);
                trial_time(itrial) = omp_get_wtime() - st;

                redirect.flush(spdlog::level::debug, "STKFMM");
            }

            spdlog::info("FMM startup + evalulation time: {}", trial_time.mean());
        }

        // Now do the real evaluation
        kernels::Evaluator stresslet_kernel_main;
        if (params.pair_evaluator == "FMM") {
            const int mult_order = params.stkfmm.periphery_stresslet_multipole_order;
            const int max_pts = params.stkfmm.periphery_stresslet_max_points;
            utils::LoggerRedirect redirect(std::cout);
            stresslet_kernel_main = kernels::FMM<stkfmm::Stk3DFMM>(mult_order, max_pts, stkfmm::PAXIS::NONE,
                                                                   stkfmm::KERNEL::PVel, kernels::stokes_pvel_fmm);
            redirect.flush(spdlog::level::debug, "STKFMM");
        } else if (params.pair_evaluator == "CPU") {
            stresslet_kernel_main = kernels::stresslet_direct_cpu;
        } else if (params.pair_evaluator == "GPU") {
            stresslet_kernel_main = kernels::stresslet_direct_gpu;
        }

        // Keep track of a final velocity because
        Eigen::MatrixXd outer_vel;
        Eigen::VectorXd trial_time_evaluate = Eigen::VectorXd::Zero(ntrials);
        for (int itrial = 0; itrial < ntrials; itrial++) {
            utils::LoggerRedirect redirect(std::cout);

            double st = omp_get_wtime();
            Eigen::MatrixXd inner_vel = stresslet_kernel_main(r_sl, shell->node_pos_, r_trg, f_sl, f_dl, eta);
            trial_time_evaluate(itrial) = omp_get_wtime() - st;

            redirect.flush(spdlog::level::debug, "STKFMM");
            outer_vel = inner_vel;
        }

        spdlog::info("FMM evalulation time: {}", trial_time_evaluate.mean());

        std::ostringstream velstream;
        velstream << outer_vel;
        spdlog::info("vel:\n{}", velstream.str());

    } catch (const std::runtime_error &e) {
        spdlog::critical(std::string("Fatal exception caught: ") + e.what());
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Teardown MPI
    MPI_Finalize();
    return EXIT_SUCCESS;
}
