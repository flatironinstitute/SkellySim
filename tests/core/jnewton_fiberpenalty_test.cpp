/// \file unit_test_fiber_base.cpp
/// \brief Unit tests for FiberChebyshevPenalty class

// C++ includes
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// External includes
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

// skelly includes
#include <fiber_chebyshev_penalty_autodiff.hpp>
#include <skelly_sim.hpp>

// Set up some using directives to make calling things easier...
using namespace skelly_fiber;

// Need something akin to the state of the system so that we can query it for global variables from outside
// functions
// Namespace fiber_chebyshev_test = fct
namespace fct {
int N_;
double dt_;
double zeta_;
double length_;
double max_time_;
} // namespace fct

// JNewton solve by hand...
template <typename VecT>
VecT JNewtonSingleFiberPenalty(const VecT &XX, FiberChebyshevPenaltyAutodiff<VecT> &FS, const VecT &oldXX) {
    // Need a non-const version of XX so that we can actually perform the autodiff
    VecT X = XX;
    // VecT XXP = oldXX;

    // Create the evaluation vector
    VecT Y;

    // Try to push a jacobian through this
    Eigen::MatrixXd J = autodiff::jacobian(skelly_fiber::SheerDeflectionObjective<VecT>, autodiff::wrt(X),
                                           autodiff::at(X, FS, oldXX, fct::length_, fct::zeta_, fct::dt_), Y);

    // Directly form the inverse of the jacobian
    VecT dY = J.inverse() * Y;

    // Return the change after a single newton iteration
    return XX - dY;
}

// timestepping function
template <typename VecT>
std::tuple<VecT, VecT> UpdateSingleNewtonBackwardEuler(const VecT &XX, FiberChebyshevPenaltyAutodiff<VecT> &FS) {
    // Make a copy of the 'old' vector for this operation
    VecT oldXX = XX;

    // Get the new XX values...
    VecT newXX = JNewtonSingleFiberPenalty<VecT>(XX, FS, oldXX);

    // Return a tuple of both the XXnew and oldXX values...
    return std::make_tuple(newXX, oldXX);
}

// Create a fiber and run and track the results (do this outside of main for cleanliness and comparisons)
void run_and_track_singlenewton() {
    // Fancy printing in eigen
    Eigen::IOFormat ColumnAsRowFmt(Eigen::StreamPrecision, 0, ",", ",", "", "", "[", "]");

    // Figure out the timesteps we are doing
    int nsteps = int(fct::max_time_ / fct::dt_);

    // Create a fiber solver and initial state
    // NOTE: This is what sets the type(s) for everything...
    auto [FS, XX] = SetupSolverInitialstate<autodiff::VectorXreal>(fct::N_, fct::length_);
    autodiff::VectorXreal oldXX = XX;
    double t = 0.0;

    // Redirect everything through spdlog
    spdlog::info("---- Initial state ----");
    std::ostringstream initial_fs_stream;
    initial_fs_stream << FS;
    spdlog::info(initial_fs_stream.str());
    std::ostringstream initial_xx_stream;
    initial_xx_stream << XX.format(ColumnAsRowFmt);
    spdlog::info(initial_xx_stream.str());

    // Run the timestepping
    for (auto i = 0; i < nsteps; i++) {
        spdlog::info("Timestep: {}; of {}", i, nsteps);

        // Time the solve
        double st = omp_get_wtime();
        std::tie(XX, oldXX) = UpdateSingleNewtonBackwardEuler(XX, FS);
        double et = omp_get_wtime() - st;

        // Update 'time'
        t += fct::dt_;

        // Get the error in the extensibility
        auto [XC, YC, TC, ext_err] =
            skelly_fiber::Extricate<autodiff::VectorXreal::Scalar, autodiff::VectorXreal>(XX, FS, fct::length_);

        // Write out some information
        spdlog::info("... Runtime:              {}", et);
        spdlog::info("... Extensibility Error:  {}", ext_err.val());
        std::ostringstream xc_stream;
        xc_stream << "XC = " << XC.format(ColumnAsRowFmt);
        spdlog::info(xc_stream.str());
        std::ostringstream yc_stream;
        yc_stream << "YC = " << YC.format(ColumnAsRowFmt);
        spdlog::info(yc_stream.str());
        std::ostringstream tc_stream;
        tc_stream << "TC = " << TC.format(ColumnAsRowFmt);
        spdlog::info(tc_stream.str());
    }
}

// Try to write a solver for the a chebyshev fiber in the sheer deflection flow based on how we solve the full system
int main(int argc, char *argv[]) {
    // Set up some using directives to make calling things easier...
    using namespace skelly_fiber;

    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);

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

    // Wrap everything in a try/catch block to see if something has gone wrong
    try {
        // Set the fake parameters for our run
        fct::N_ = 20;
        fct::length_ = 1.0;
        fct::zeta_ = 1000.0;
        fct::dt_ = 1.0 / fct::zeta_ / 4.0;
        fct::max_time_ = 3.0 / fct::zeta_;

        // Call our run function
        run_and_track_singlenewton();

    } catch (const std::runtime_error &e) {
        spdlog::critical(std::string("Fatal exception caught: ") + e.what());
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Teardown MPI
    MPI_Finalize();
    return EXIT_SUCCESS;
}
