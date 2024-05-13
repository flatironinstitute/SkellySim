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
#include <kernels.hpp>
#include <utils.hpp>

using Eigen::MatrixXd;

// Create a timing result structure so we can measure things
class HydroTimingResult {
  public:
    Eigen::VectorXd runtimes_;
    Eigen::VectorXd err_;
    Eigen::VectorXd rel_err_;
    Eigen::VectorXd fmm_setup_;

    HydroTimingResult() = default;

    HydroTimingResult(CVectorRef &runtimes, CVectorRef &err, CVectorRef &rel_err, CVectorRef &fmm_setup)
        : runtimes_(runtimes), err_(err), rel_err_(rel_err), fmm_setup_(fmm_setup) {}
};

// Stokeslet solve, input a string for the driver
HydroTimingResult run_and_track_stokeslet(const std::string &driver, const int mult_order, const int max_pts,
                                          const int n_src, const int n_trg, const int ntrials) {
    // Setup random matrices to test against
    const double eta = 1.3;

    // Make these take up slightly more space
    MatrixXd r_src = MatrixXd::Random(3, n_src);
    MatrixXd r_trg = MatrixXd::Random(3, n_trg);
    MatrixXd nullmat;

    // Generate the "truth" value
    MatrixXd f_src = MatrixXd::Random(3, n_src);
    MatrixXd ref = kernels::stokeslet_direct_cpu(r_src, nullmat, r_trg, f_src, nullmat, eta);

    // Vector of results to keep track of...
    Eigen::VectorXd runtimes = Eigen::VectorXd::Zero(ntrials);
    Eigen::VectorXd err = Eigen::VectorXd::Zero(ntrials);
    Eigen::VectorXd rel_err = Eigen::VectorXd::Zero(ntrials);
    Eigen::VectorXd fmm_setup = Eigen::VectorXd::Zero(ntrials);

    // If we are doing the FMM, measure the startup+evaluation cost (it sets the tree the first time)
    if (driver == "FMM") {
        for (int i = 0; i < ntrials; i++) {
            double st = omp_get_wtime();
            auto stokeslet_kernel_fmm = kernels::FMM<stkfmm::Stk3DFMM>(mult_order, max_pts, stkfmm::PAXIS::NONE,
                                                                       stkfmm::KERNEL::Stokes, kernels::stokes_vel_fmm);
            Eigen::MatrixXd other = stokeslet_kernel_fmm(r_src, nullmat, r_trg, f_src, nullmat, eta);
            double et = omp_get_wtime();

            fmm_setup(i) = et - st;
        }
    }

    // Now run the actual result
    kernels::Evaluator stokeslet_kernel_eval;
    if (driver == "FMM") {
        utils::LoggerRedirect redirect(std::cout);
        stokeslet_kernel_eval = kernels::FMM<stkfmm::Stk3DFMM>(mult_order, max_pts, stkfmm::PAXIS::NONE,
                                                               stkfmm::KERNEL::Stokes, kernels::stokes_vel_fmm);
        redirect.flush(spdlog::level::debug, "STKFMM");
    } else if (driver == "CPU") {
        stokeslet_kernel_eval = kernels::stokeslet_direct_cpu;
    } else if (driver == "GPU") {
        stokeslet_kernel_eval = kernels::stokeslet_direct_gpu;
    }

    // Run the number of iterations to get an idea of what the timing is...
    for (int i = 0; i < ntrials; ++i) {
        // Timing information is now consistent (control openmp from outside)
        double st = omp_get_wtime();
        Eigen::MatrixXd val = stokeslet_kernel_eval(r_src, nullmat, r_trg, f_src, nullmat, eta);
        double et = omp_get_wtime();

        // Save timing information and the error
        runtimes(i) = et - st;
        err(i) = sqrt((val - ref).squaredNorm());

        // Do a relative error as well, do the norm of the two matrix difference, minus the norm of the first
        rel_err(i) = (val - ref).norm() / ref.norm();

        // TODO(cje) remove this, print out the two matrices and inspect them...
        // Eigen::IOFormat ColumnAsRowFmt(Eigen::StreamPrecision, 0, ",", ",", "", "", "[", "]");
        std::ostringstream refstream;
        refstream << ref;
        spdlog::info("ref:\n{}", refstream.str());
        std::ostringstream valstream;
        valstream << val;
        spdlog::info("val:\n{}", valstream.str());
    }

    return HydroTimingResult(runtimes, err, rel_err, fmm_setup);
}

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

    int ntrials;
    int mult_order;
    int max_pts;
    int n_src_max;
    int n_trg_max;
    std::string driver;

    Teuchos::CommandLineProcessor cmdp(false, true);
    cmdp.setOption("ntrials", &ntrials, "number of iterations", true);
    cmdp.setOption("mult_order", &mult_order, "multipole_order", true);
    cmdp.setOption("max_pts", &max_pts, "max points", true);
    cmdp.setOption("n_src_max", &n_src_max, "log_10 maximum number of sources", true);
    cmdp.setOption("n_trg_max", &n_trg_max, "log_10 maximum number of targets", true);
    cmdp.setOption("driver", &driver, "driver [CPU,GPU,FMM]", true);

    if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // This program is not designed to run with MPI, so if we have more than 1 mpi rank, throw an error and exit
    if (mpisize > 1) {
        spdlog::error("This program was not designed for use with MPI, which will overcount the sources.");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    std::vector<int> n_src_vec;
    std::vector<int> n_trg_vec;
    std::vector<std::string> kernels{"stokeslet"};
    std::vector<std::string> drivers;
    // Build the configurations we are going to run
    // i is the exponent for the base 10
    for (int i = 1; i <= n_src_max; i++) {
        n_src_vec.push_back(std::pow(10.0, i));
    }
    for (int i = 1; i <= n_trg_max; i++) {
        n_trg_vec.push_back(std::pow(10.0, i));
    }
    drivers.push_back(driver);

    // Create a result framework to stash the timing information
    // all_results[driver][kernel][n_src][n_trg]
    std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::unordered_map<int, std::unordered_map<int, HydroTimingResult>>>>
        all_results;

    try {
        // Setup our runs...
        for (auto driver : drivers) {
            for (auto kernel : kernels) {
                for (auto n_src : n_src_vec) {
                    for (auto n_trg : n_trg_vec) {
                        spdlog::info("----------------------------------------------------------------");
                        spdlog::info("Performance testing kernel: {}, driver: {}, n_src: {}, n_trg: {}, ntrials: {}",
                                     kernel, driver, n_src, n_trg, ntrials);

                        all_results[driver][kernel][n_src][n_trg] =
                            run_and_track_stokeslet(driver, mult_order, max_pts, n_src, n_trg, ntrials);

                        // Print out just the timing result so we're not bored
                        auto mresult = all_results[driver][kernel][n_src][n_trg];
                        spdlog::info("...(mean) runtime: {}", mresult.runtimes_.mean());
                    }
                }
            }
        }

        // Print out something at the end
        Eigen::IOFormat ColumnAsRowFmt(Eigen::StreamPrecision, 0, ",", ",", "", "", "[", "]");
        spdlog::info("");
        spdlog::info("");
        spdlog::info("-------- Results --------");

        for (auto driver : drivers) {
            for (auto kernel : kernels) {
                for (auto n_src : n_src_vec) {
                    for (auto n_trg : n_trg_vec) {
                        auto mresult = all_results[driver][kernel][n_src][n_trg];

                        spdlog::info("--------------------------------");
                        spdlog::info("driver: {}, kernel: {}, n_src: {}, n_trg: {}", driver, kernel, n_src, n_trg);

                        spdlog::info("...(mean) runtime: {}, fmm_setup: {}, err: {}, rel_err: {}",
                                     mresult.runtimes_.mean(), mresult.fmm_setup_.mean(), mresult.err_.mean(),
                                     mresult.rel_err_.mean());
                    }
                }
            }
        }

    } catch (const std::runtime_error &e) {
        spdlog::critical(std::string("Fatal exception caught: ") + e.what());
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Teardown MPI
    MPI_Finalize();
    return EXIT_SUCCESS;
}
