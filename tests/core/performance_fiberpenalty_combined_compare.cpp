/// \file performance_fiberpenalty_combined_compare.cpp
/// \brief Testing the performance of various implementations of a solver for the FiberChebyshevPenaltyAutodiff class

// C++ includes
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// External includes
#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosTpetraAdapter.hpp>
#include <Kokkos_Core.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_Vector.hpp>
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

// skelly includes
#include <fiber_chebyshev_penalty_autodiff.hpp>
#include <fiber_chebyshev_penalty_autodiff_external.hpp>
#include <skelly_chebyshev.hpp>
#include <skelly_sim.hpp>
#include <utils.hpp>

// Set up some using directives to make calling things easier...
using namespace skelly_fiber;

// Result data structure for our runs to keep track of
class FiberTimingResult {
  public:
    std::string solver_;
    std::string autodiffvar_;
    Eigen::VectorXd runtimes_;
    Eigen::VectorXd max_extensibility_error_;
    Eigen::VectorXd deflection_;

    FiberTimingResult() = default;

    FiberTimingResult(const std::string &solver, const std::string &autodiffvar, CVectorRef &runtimes,
                      CVectorRef &max_extensibility_error, CVectorRef &deflection)
        : solver_(solver), autodiffvar_(autodiffvar), runtimes_(runtimes),
          max_extensibility_error_(max_extensibility_error), deflection_(deflection) {}
};

/// @brief Helper function to construct a FiberSolver of this type and an initial state vector
///
/// Note: This is specific to an initial condition and the penalty autodiff types. This version should be general for
/// all new FiberSolver types
template <class FiberT>
auto SetupSolverInitialstateGeneral(const int N, const double L) {
    // Get the typedefs from the FiberT
    using VecT = typename FiberT::vector_type;
    // Set up the numbers of things that are going on
    int NT = N - 2;
    int Neq = N - 4;
    int NTeq = NT - 2;

    // Create the fiber solver
    FiberT FS = FiberT(N, NT, Neq, NTeq);

    // Now create our fiber in XYT
    VecT init_X = Eigen::VectorXd::Zero(FS.n_nodes_);
    VecT init_Y = Eigen::VectorXd::Zero(FS.n_nodes_);
    VecT init_T = Eigen::VectorXd::Zero(FS.n_nodes_tension_);
    init_Y[init_Y.size() - 1 - 3] = L / 2.0;
    init_Y[init_Y.size() - 1 - 2] = 1.0;
    VecT XX(init_X.size() + init_Y.size() + init_T.size());
    XX << init_X, init_Y, init_T;

    // return a tuple of these
    return std::make_tuple(FS, XX);
}

/// @brief Extricate the 'XC' like variables and the extensibility error
template <class FiberT, typename VecT>
std::tuple<VecT, VecT, VecT, double> ExtricateGeneral(const VecT &XX, FiberT &FS, const double L) {
    throw std::runtime_error("Should not be calling base version of SheerDeflectionObjectiveGeneral");
    return std::make_tuple<VecT, VecT, VecT, double>(VecT::Zero(1), VecT::Zero(1), VecT::Zero(1), 1.0);
}

/// @brief Extricate the 'XC' like variables and the extensibility error
template <>
std::tuple<autodiff::VectorXreal, autodiff::VectorXreal, autodiff::VectorXreal, double>
ExtricateGeneral(const autodiff::VectorXreal &XX, FiberChebyshevPenaltyAutodiffExternal<autodiff::VectorXreal> &FS,
                 const double L) {
    // Get the components of XX
    auto Div = DivideAndConstructFCPA(XX, L, FS.n_nodes_, FS.n_nodes_tension_, FS.n_equations_, FS.n_equations_tension_,
                                      FS.IM_, FS.IMT_);

    auto ext_error = skelly_fiber::ExtensibilityError<autodiff::real, autodiff::VectorXreal>(Div.XsC_, Div.YsC_);
    double ret_err = ext_error.val();
    return std::make_tuple(Div.XC_, Div.YC_, Div.TC_, ret_err);
}

/// @brief Extricate the 'XC' like variables and the extensibility error
template <>
std::tuple<autodiff::VectorXreal, autodiff::VectorXreal, autodiff::VectorXreal, double>
ExtricateGeneral(const autodiff::VectorXreal &XX, FiberChebyshevPenaltyAutodiff<autodiff::VectorXreal> &FS,
                 const double L) {
    // Get the components of XX
    auto Div = FS.DivideAndConstruct(XX, L);

    auto ext_error = skelly_fiber::ExtensibilityError<autodiff::real, autodiff::VectorXreal>(Div.XsC_, Div.YsC_);
    double ret_err = ext_error.val();
    return std::make_tuple(Div.XC_, Div.YC_, Div.TC_, ret_err);
}

template <class FiberT, typename VecT>
VecT SheerDeflectionObjectiveGeneral(const VecT &XX, FiberT &FS, const VecT &oldXX, const double L, const double zeta,
                                     const double dt) {
    throw std::runtime_error("Should not be calling base versio of SheerDeflectionObjectiveGeneral");
    return VecT::Zero(1);
}

// Template specialization to get around the fact that some things are different...
template <>
autodiff::VectorXreal
SheerDeflectionObjectiveGeneral<FiberChebyshevPenaltyAutodiffExternal<autodiff::VectorXreal>, autodiff::VectorXreal>(
    const autodiff::VectorXreal &XX, FiberChebyshevPenaltyAutodiffExternal<autodiff::VectorXreal> &FS,
    const autodiff::VectorXreal &oldXX, const double L, const double zeta, const double dt) {
    // Set up a cute 'using' to get the internal scalar type of VecT
    using VecT = typename autodiff::VectorXreal;
    using ScalarT = typename autodiff::VectorXreal::Scalar;

    auto Div = DivideAndConstructFCPA(XX, L, FS.n_nodes_, FS.n_nodes_tension_, FS.n_equations_, FS.n_equations_tension_,
                                      FS.IM_, FS.IMT_);
    auto oDiv = DivideAndConstructFCPA(oldXX, L, FS.n_nodes_, FS.n_nodes_tension_, FS.n_equations_,
                                       FS.n_equations_tension_, FS.IM_, FS.IMT_);

    // Get the forces
    auto [FxC, FyC, AFxC, AFyC] = FiberForces(Div, oDiv, L, FS.n_equations_);

    // Sheer velocity
    VecT UC = zeta * Div.YC_;
    VecT VC = VecT::Zero(Div.YC_.size());
    VecT UsC = zeta * Div.YsC_;
    VecT VsC = VecT::Zero(Div.YsC_.size());
    VecT oUsC = zeta * oDiv.YsC_;
    VecT oVsC = VecT::Zero(oDiv.YsC_.size());

    // Get the evolution equations
    auto [teqXC, teqYC] = FiberEvolution(AFxC, AFyC, Div, oDiv, UC, VC, dt);

    // Get the tension equation
    auto teqTC = FiberPenaltyTension(Div, oDiv, UsC, VsC, oUsC, oVsC, dt, FS.n_equations_tension_);

    // Get the boundary conditions
    VecT cposition{{0.0, 0.0}};
    VecT cdirector{{0.0, 1.0}};
    FiberBoundaryCondition<ScalarT> BCL = ClampedBC<ScalarT, VecT>(Div, oDiv, FSIDE::left, cposition, cdirector);
    FiberBoundaryCondition<ScalarT> BCR = FreeBC<ScalarT, VecT>(Div, FSIDE::right);

    // Combine together boundary conditions with the equations
    auto eqXC = CombineXWithBCs<ScalarT, VecT>(teqXC, BCL, BCR);
    auto eqYC = CombineYWithBCs<ScalarT, VecT>(teqYC, BCL, BCR);
    auto eqTC = CombineTPenaltyWithBCs<ScalarT, VecT>(teqTC, BCL, BCR);

    VecT eq_full(eqXC.size() + eqYC.size() + eqTC.size());
    eq_full << eqXC, eqYC, eqTC;
    return eq_full;
}

// Template specialization to get around the fact that some things are different...
template <>
autodiff::VectorXreal
SheerDeflectionObjectiveGeneral<FiberChebyshevPenaltyAutodiff<autodiff::VectorXreal>, autodiff::VectorXreal>(
    const autodiff::VectorXreal &XX, FiberChebyshevPenaltyAutodiff<autodiff::VectorXreal> &FS,
    const autodiff::VectorXreal &oldXX, const double L, const double zeta, const double dt) {
    // Set up a cute 'using' to get the internal scalar type of VecT
    using VecT = typename autodiff::VectorXreal;
    using ScalarT = typename autodiff::VectorXreal::Scalar;

    auto Div = FS.DivideAndConstruct(XX, L);
    auto oDiv = FS.DivideAndConstruct(oldXX, L);

    // Get the forces
    auto [FxC, FyC, AFxC, AFyC] = FiberForces(Div, oDiv, L, FS.n_equations_);

    // Sheer velocity
    VecT UC = zeta * Div.YC_;
    VecT VC = VecT::Zero(Div.YC_.size());
    VecT UsC = zeta * Div.YsC_;
    VecT VsC = VecT::Zero(Div.YsC_.size());
    VecT oUsC = zeta * oDiv.YsC_;
    VecT oVsC = VecT::Zero(oDiv.YsC_.size());

    // Get the evolution equations
    auto [teqXC, teqYC] = FiberEvolution(AFxC, AFyC, Div, oDiv, UC, VC, dt);

    // Get the tension equation
    auto teqTC = FiberPenaltyTension(Div, oDiv, UsC, VsC, oUsC, oVsC, dt, FS.n_equations_tension_);

    // Get the boundary conditions
    VecT cposition{{0.0, 0.0}};
    VecT cdirector{{0.0, 1.0}};
    FiberBoundaryCondition<ScalarT> BCL = ClampedBC<ScalarT, VecT>(Div, oDiv, FSIDE::left, cposition, cdirector);
    FiberBoundaryCondition<ScalarT> BCR = FreeBC<ScalarT, VecT>(Div, FSIDE::right);

    // Combine together boundary conditions with the equations
    auto eqXC = CombineXWithBCs<ScalarT, VecT>(teqXC, BCL, BCR);
    auto eqYC = CombineYWithBCs<ScalarT, VecT>(teqYC, BCL, BCR);
    auto eqTC = CombineTPenaltyWithBCs<ScalarT, VecT>(teqTC, BCL, BCR);

    VecT eq_full(eqXC.size() + eqYC.size() + eqTC.size());
    eq_full << eqXC, eqYC, eqTC;
    return eq_full;
}

// ****************************************************************************
// Single step Newton solve
// ****************************************************************************
// Directly computes the Jacobian inverse to solve the linearized system

// Single step of a Newton solve by hand, using the inverse of the Jacobian
template <class FiberT, typename VecT>
VecT JNewtonSingleFiberPenaltyGeneral(const VecT &XX, FiberT &FS, const VecT &oldXX, const double L, const double zeta,
                                      const double dt) {
    // Need a non-const version of XX so that we can actually perform the autodiff
    VecT X = XX;

    // Create the evaluation vector
    VecT Y;

    // Try to push a jacobian through this
    Eigen::MatrixXd J = autodiff::jacobian(SheerDeflectionObjectiveGeneral<FiberT, VecT>, autodiff::wrt(X),
                                           autodiff::at(X, FS, oldXX, L, zeta, dt), Y);

    // Directly form the inverse of the jacobian
    VecT dY = J.inverse() * Y;

    // Return the change after a single newton iteration
    return XX - dY;
}

// Timestepping function for the newton version of the solve
template <class FiberT, typename VecT>
std::tuple<VecT, VecT> UpdateSingleNewtonBackwardEulerGeneral(const VecT &XX, FiberT &FS, const double L,
                                                              const double zeta, const double dt) {
    // Make a copy of the 'old' vector for this operation
    VecT oldXX = XX;

    // Get the new XX values...
    VecT newXX = JNewtonSingleFiberPenaltyGeneral<FiberT, VecT>(XX, FS, oldXX, L, zeta, dt);

    // Return a tuple of both the XXnew and oldXX values...
    std::tuple<VecT, VecT> ret_tuple(newXX, oldXX);
    return ret_tuple;
    // return std::make_tuple<VecT, VecT>(newXX, oldXX);
}

// Create a fiber and run and track the results (do this outside of main for cleanliness and comparisons)
template <class FiberT>
FiberTimingResult run_and_track_singlenewton(const std::string &solver, const std::string &autodiffvar, const int N,
                                             const double max_time, const double dt, const double L,
                                             const double zeta) {
    // Cute using to get the scalar type of the vector
    using VecT = typename FiberT::vector_type;
    using ScalarT = typename VecT::Scalar;
    // Fancy printing in eigen
    Eigen::IOFormat ColumnAsRowFmt(Eigen::StreamPrecision, 0, ",", ",", "", "", "[", "]");

    // Figure out the timesteps we are doing
    int nsteps = static_cast<int>(max_time / dt);

    // Create a fiber solver and initial state
    // NOTE: This is what sets the type(s) for everything...
    auto [FS, XX] = SetupSolverInitialstateGeneral<FiberT>(N, L);
    VecT oldXX = XX;
    double t = 0.0;

    // Redirect everything through spdlog
    spdlog::info("---- Initial state ----");
    std::ostringstream initial_fs_stream;
    initial_fs_stream << FS;
    spdlog::info(initial_fs_stream.str());
    std::ostringstream initial_xx_stream;
    initial_xx_stream << XX.format(ColumnAsRowFmt);
    spdlog::info(initial_xx_stream.str());

    Eigen::VectorXd runtimes = Eigen::VectorXd::Zero(nsteps);
    Eigen::VectorXd max_extensibility_error = Eigen::VectorXd::Zero(nsteps);
    Eigen::VectorXd deflection = Eigen::VectorXd::Zero(nsteps);

    // Run the timestepping
    for (auto i = 0; i < nsteps; i++) {
        spdlog::info("Timestep: {}; of {}", i, nsteps);

        // Time the solve
        double st = omp_get_wtime();
        std::tie(XX, oldXX) = UpdateSingleNewtonBackwardEulerGeneral<FiberT, VecT>(XX, FS, L, zeta, dt);
        double et = omp_get_wtime() - st;

        // Update 'time'
        t += dt;

        // Get the error in the extensibility
        auto [XC, YC, TC, ext_err] = ExtricateGeneral<FiberT, VecT>(XX, FS, 1.0);

        // Write out some information
        spdlog::info("... Runtime:              {}", et);
        spdlog::info("... Extensibility Error:  {}", ext_err);

        // Save information
        runtimes(i) = et;
        max_extensibility_error(i) = ext_err;
        auto deflection_real = skelly_chebyshev::RightEvalPoly<ScalarT, VecT>(XC);

        deflection(i) = deflection_real.val();
    }

    return FiberTimingResult(solver, autodiffvar, runtimes, max_extensibility_error, deflection);
}

// Try to write a solver for the a chebyshev fiber in the sheer deflection flow based on how we solve the full
// system
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

    // Set the control variables for this simulation
    int N;
    double dt;
    std::string fibertype;
    std::string solver;
    std::string autodiffvar;

    // Get command line arguments
    Teuchos::CommandLineProcessor cmdp(false, true);
    cmdp.setOption("N", &N, "number of fiber points", true);
    cmdp.setOption("dt", &dt, "timestep size", true);
    cmdp.setOption("fibertype", &fibertype, "fiber type [autodiff_external]", true);
    cmdp.setOption("solver", &solver, "solver [jnewton,belos,all]", true);
    cmdp.setOption("autodiff", &autodiffvar, "autodiff var [real,dual]", true);

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

    double tolerance = 1e-10;
    double master_zeta = 1000.0;
    double max_time = 3.0 / master_zeta;
    double master_length = 1.0;

    // Create a result that we can use
    // ["fibertype"]["solver"]["autodiffvar"][N][dt]
    std::unordered_map<
        std::string,
        std::unordered_map<
            std::string,
            std::unordered_map<std::string, std::unordered_map<double, std::unordered_map<double, FiberTimingResult>>>>>
        all_results;

    // Wrap everything in a try/catch block to see if something has gone wrong
    try {
        spdlog::info("----------------------------------------------------------------");
        spdlog::info("Working on N = {}, dt = {}, fiber type = {}, solver = {}, autodiff = {}", N, dt, fibertype,
                     solver, autodiffvar);

        // Very ugly ifelse block for all the options we have
        if (fibertype == "autodiff_external") {
            if (solver == "jnewton") {
                if (autodiffvar == "real") {
                    all_results[fibertype][solver][autodiffvar][N][dt] =
                        run_and_track_singlenewton<FiberChebyshevPenaltyAutodiffExternal<autodiff::VectorXreal>>(
                            solver, autodiffvar, N, max_time, dt, master_length, master_zeta);
                }
            }
        } else if (fibertype == "autodiff") {
            if (solver == "jnewton") {
                if (autodiffvar == "real") {
                    all_results[fibertype][solver][autodiffvar][N][dt] =
                        run_and_track_singlenewton<FiberChebyshevPenaltyAutodiff<autodiff::VectorXreal>>(
                            solver, autodiffvar, N, max_time, dt, master_length, master_zeta);
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
