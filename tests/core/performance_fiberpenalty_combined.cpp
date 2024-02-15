/// \file performance_fiberpenalty_combined.cpp
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
#include <skelly_chebyshev.hpp>
#include <skelly_sim.hpp>
#include <utils.hpp>

// Set up some using directives to make calling things easier...
using namespace skelly_fiber;

// Need something akin to the state of the system so that we can query it for global variables from outside
// functions. This includes a Fiber, as we need to be able to access it from outside. This includes a Fiber, as we need
// to be able to access it from outside Namespace fiber_chebyshev_test = fct
//
// JX0 is the jacobian at point X0 for use in linear algebra solutions with Belos
namespace fct {
int N_;
double dt_;
double zeta_;
double length_;
double max_time_;
double tolerance_;

// Fiber that we keep track of
FiberChebyshevPenaltyAutodiff<autodiff::VectorXreal> FS_;

// The Jacobian at state X0
Eigen::MatrixXd JX0_;
// The evalulation of the objective function at X0
Eigen::VectorXd Y0_;

// Current solution
Eigen::VectorXd curr_solution_;
} // namespace fct

// Result data structure for our runs to keep track of
template <typename VecT>
class TimingResult {
  public:
    std::vector<double> runtimes_;
    std::vector<double> ts_;
    std::vector<double> max_extensibility_error_;
    std::vector<double> deflection_;
    std::vector<VecT> xx_;

    TimingResult() = default;

    TimingResult(const std::vector<double> &runtimes, const std::vector<double> &ts,
                 const std::vector<double> &max_extensibility_error, const std::vector<double> &deflection,
                 const std::vector<VecT> &xx)
        : runtimes_(runtimes), ts_(ts), max_extensibility_error_(max_extensibility_error), deflection_(deflection),
          xx_(xx) {}
};

// ****************************************************************************
// Belos information and tests
// ****************************************************************************

// Test for just the get_solution_map calls
std::tuple<VectorMap> get_fiber_solution_map(double *x) {
    using Eigen::Map;
    using Eigen::VectorXd;

    auto fib_sol_size = fct::FS_.get_local_solution_size();
    return std::make_tuple(VectorMap(x, fib_sol_size));
}
// Test for just the get_solution_map calls
std::tuple<CVectorMap> get_fiber_solution_map(const double *x) {
    using Eigen::Map;
    using Eigen::VectorXd;

    auto fib_sol_size = fct::FS_.get_local_solution_size();
    return std::make_tuple(CVectorMap(x, fib_sol_size));
}

// Get the jacobian at point X0 and store it for use, along with F(X0)
template <typename VecT>
std::tuple<Eigen::MatrixXd, Eigen::VectorXd> ConstructJacobianAndY0(const VecT &X0) {
    // We have to autodiff to get the evaluation and the jacobian
    VecT X = X0;
    VecT Y;

    // Push an autodiff through the function, using the argument X0 for the old state (X should get the new state)
    Eigen::MatrixXd J = autodiff::jacobian(skelly_fiber::SheerDeflectionObjective<VecT>, autodiff::wrt(X),
                                           autodiff::at(X, fct::FS_, X0, fct::length_, fct::zeta_, fct::dt_), Y);

    // XXX I hate this, but coerce the values into a VectorXd to return
    Eigen::VectorXd Y0(Y.size());
    for (auto i = 0; i < Y.size(); i++) {
        Y0(i) = Y(i).val();
    }

    return std::make_tuple(J, Y0);
}

// Apply the matvec operation of the jacobian evaluated at x0 and the objective evaluated there as well
// to figure out the next step we should take
Eigen::VectorXd apply_matvec_jx0_xi(CVectorRef &xi) {
    // Get local node count and solution sizes to check for validity
    const int total_node_count = fct::FS_.get_local_node_count();
    const int sol_size = fct::FS_.get_local_solution_size();

    // Sanity check on the size of the solution versus what the fiber is doing
    assert(sol_size == xi.size());

    // Return vector
    Eigen::VectorXd res(sol_size);

    // Get the mapping between the input vector and where we stick the fiber/whatever information
    auto [x_fibers] = get_fiber_solution_map(xi.data());
    auto [res_fibers] = get_fiber_solution_map(res.data());

    // Do the single fiber matvec operation
    res_fibers = fct::JX0_ * x_fibers;

    return res;
}

// Create a TPetra operator for the jacobian that mirrors what skellysim already does
// Performs the operation of J(X0) * Xi
// FCPA = FiberChebyshevPenaltyAutodiff
class FCPA_JacobianMultiply : public Tpetra::Operator<> {
  public:
    // Tpetra::Operator subclasses should always define these four typedefs.
    typedef Tpetra::Operator<>::scalar_type scalar_type;
    typedef Tpetra::Operator<>::local_ordinal_type local_ordinal_type;
    typedef Tpetra::Operator<>::global_ordinal_type global_ordinal_type;
    typedef Tpetra::Operator<>::node_type node_type;
    // The type of the input and output arguments of apply().
    typedef Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type> MV;
    // The Map specialization used by this class.
    typedef Tpetra::Map<local_ordinal_type, global_ordinal_type, node_type> map_type;

  private:
    // This is an implementation detail; users don't need to see it.
    typedef Tpetra::Import<local_ordinal_type, global_ordinal_type, node_type> import_type;

  public:
    // comm: The communicator over which to distribute those rows and columns.
    FCPA_JacobianMultiply(const Teuchos::RCP<const Teuchos::Comm<int>> comm) : comm_(comm), rank_(comm->getRank()) {
        TEUCHOS_TEST_FOR_EXCEPTION(comm.is_null(), std::invalid_argument,
                                   "FCPA_JacobianMultiply constructor: The input Comm object must be nonnull.");
        const global_ordinal_type indexBase = 0;
        const auto fiber_sol_size = fct::FS_.get_local_solution_size();
        const int local_size = fiber_sol_size;

        // Construct a map for our block row distribution
        opMap_ =
            rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), local_size, indexBase, comm));
    }
    virtual ~FCPA_JacobianMultiply() {}
    Teuchos::RCP<const map_type> getDomainMap() const { return opMap_; };
    Teuchos::RCP<const map_type> getRangeMap() const { return opMap_; };

    // Apply is the guts of where we force our Jacobian multiplication into the Tpetra operator
    void apply(const MV &X, MV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
               scalar_type alpha = Teuchos::ScalarTraits<scalar_type>::one(),
               scalar_type beta = Teuchos::ScalarTraits<scalar_type>::zero()) const {
        // Load the information into the operator
        for (size_t c = 0; c < X.getNumVectors(); ++c) {
            CVectorMap x_local(X.getData(c).getRawPtr(), X.getLocalLength());
            VectorMap res(Y.getDataNonConst(c).getRawPtr(), Y.getLocalLength());
            res = apply_matvec_jx0_xi(x_local);
        }
    }

  private:
    Teuchos::RCP<const map_type> opMap_;
    Teuchos::RCP<const Teuchos::Comm<int>> comm_;
    const int rank_;
};

// Build a solver for the problem at hand. This really just takes in a Tpetra operator and then evaluates it.
// The magic on the inside is that we set our matvec_T operation to the Tpetra solver above that uses our
// own jacobian/RHS to solve the system of equations, so it gets stuffed in there.
template <typename matvec_T>
class FCPA_Solver {
  public:
    typedef typename matvec_T::scalar_type ST;
    typedef Tpetra::Vector<typename matvec_T::scalar_type, typename matvec_T::local_ordinal_type,
                           typename matvec_T::global_ordinal_type, typename matvec_T::node_type>
        SV;
    typedef typename matvec_T::MV MV;
    typedef Tpetra::Operator<ST> OP;

    FCPA_Solver() {
        comm_ = Tpetra::getDefaultComm();
        matvec_ = rcp(new matvec_T(comm_));
        map_ = matvec_->getDomainMap();
        X_ = rcp(new SV(map_));
        RHS_ = rcp(new SV(map_));
    }

    CVectorMap get_solution() { return CVectorMap(X_->getData(0).getRawPtr(), X_->getLocalLength()); }

    double get_residual() {
        Teuchos::RCP<SV> Y(new SV(map_));
        matvec_->apply(*X_, *Y);
        CVectorMap RHS_map(RHS_->getData(0).getRawPtr(), RHS_->getLocalLength());
        CVectorMap Y_map(Y->getData(0).getRawPtr(), Y->getLocalLength());
        double residual = (RHS_map - Y_map).squaredNorm();
        MPI_Allreduce(MPI_IN_PLACE, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return sqrt(residual);
    }

    // Trivially set the RHS of the equation from the global state version that we keep around...
    void set_RHS() {
        const auto fib_sol_size = fct::FS_.get_local_solution_size();
        VectorMap RHS_fib(RHS_->getDataNonConst(0).getRawPtr(), fib_sol_size);

        // Initialize the RHS vector
        // For now, this is just the Y0 term we are keeping track of
        RHS_fib = fct::Y0_;
    }

    // Solve the problem
    // We are assuming that fct::JX0_ and fct::Y0_ are set before this point
    bool solve() {
        Belos::LinearProblem<ST, MV, OP> problem(matvec_, X_, RHS_);
        problem.setProblem();

        Teuchos::ParameterList belosList;
        // allowed
        belosList.set("Convergence Tolerance", fct::tolerance_);
        belosList.set("Orthogonalization", "ICGS");
        belosList.set("Verbosity", Belos::MsgType::IterationDetails + Belos::MsgType::FinalSummary +
                                       Belos::MsgType::StatusTestDetails);
        belosList.set("Output Frequency", 1);
        belosList.set("Output Style", Belos::OutputType::General);

        // Create the solver itself
        Belos::PseudoBlockGmresSolMgr<ST, MV, OP> solver(rcpFromRef(problem), rcpFromRef(belosList));
        utils::LoggerRedirect redirect(std::cout);

        // Time this
        double st = omp_get_wtime();
        Belos::ReturnType ret = solver.solve();
        redirect.flush(spdlog::level::trace, "Belos");

        if (ret == Belos::Converged) {
            spdlog::info("... ... Belos converged");
            spdlog::info("... ... Belos iterations:     {}", solver.getNumIters());
            spdlog::info("... ... Belos time:           {}", omp_get_wtime() - st);
            spdlog::info("... ... Belos tolerance:      {}", solver.achievedTol());
        } else {
            spdlog::info("... ... Belos FAILED TO CONVERGED!");
            spdlog::info("... ... Belos iterations:     {}", solver.getNumIters());
            spdlog::info("... ... Belos time:           {}", omp_get_wtime() - st);
            spdlog::info("... ... Belos tolerance:      {}", solver.achievedTol());
        }

        return ret == Belos::Converged;
    }

  private:
    Teuchos::RCP<const Teuchos::Comm<int>> comm_;
    Teuchos::RCP<matvec_T> matvec_;
    Teuchos::RCP<SV> X_;
    Teuchos::RCP<SV> RHS_;
    Teuchos::RCP<const Tpetra::Map<>> map_;
};

// Belos timestepping scheme
template <typename VecT>
std::tuple<VecT, VecT> UpdateBelosBackwardEuler(const VecT &XX) {
    // The first thing we need to do is create the jacobian and result vector in the namespace, also the oldXX
    VecT oldXX = XX;

    // Prep the state (fct)
    double st_constructjacobian = omp_get_wtime();
    std::tie(fct::JX0_, fct::Y0_) = ConstructJacobianAndY0<VecT>(XX);
    double et_constructjacobian = omp_get_wtime() - st_constructjacobian;
    spdlog::info("... ConstructJacobian time:   {}", et_constructjacobian);

    // Build a solver?
    // XXX This should probably not be done over and over and over and over...
    FCPA_Solver<FCPA_JacobianMultiply> solver;
    solver.set_RHS();

    // Actually try to solve the system
    bool converged = solver.solve();
    fct::curr_solution_ = solver.get_solution();

    spdlog::info("... Residual:             {}", solver.get_residual());

    // Now push the fiber solution back onto a new vector for them
    VecT newXX = oldXX - fct::curr_solution_;

    return std::make_tuple(newXX, oldXX);
}

// Create a fiber and run and track the results (do this outside of main for cleanliness and comparisons)
template <typename VecT>
TimingResult<VecT> run_and_track_belos() {
    // Cute using to get the scalar type of the vector
    using ScalarT = typename VecT::Scalar;
    // Fancy printing in eigen
    Eigen::IOFormat ColumnAsRowFmt(Eigen::StreamPrecision, 0, ",", ",", "", "", "[", "]");

    // Figure out the timesteps we are doing
    int nsteps = int(fct::max_time_ / fct::dt_);

    // Create a fiber solver and initial state
    // NOTE: This is what sets the type(s) for everything...
    auto [FS, XX] = SetupSolverInitialstate<VecT>(fct::N_, fct::length_);
    VecT oldXX = XX;
    double t = 0.0;
    // NOTE: Copy FS into the test namespace so we can access it globally
    fct::FS_ = FS;

    // Redirect everything through spdlog
    spdlog::info("---- Initial state ----");
    std::ostringstream initial_fs_stream;
    initial_fs_stream << FS;
    spdlog::info(initial_fs_stream.str());
    std::ostringstream initial_xx_stream;
    initial_xx_stream << XX.format(ColumnAsRowFmt);
    spdlog::info(initial_xx_stream.str());

    // Vector of results to keep track of...
    std::vector<double> runtimes;
    std::vector<double> ts;
    std::vector<double> max_extensibility_error;
    std::vector<double> deflection;
    std::vector<VecT> xx;

    // Run the timestepping
    for (auto i = 0; i < nsteps; i++) {
        spdlog::info("Timestep: {}; of {}", i, nsteps);

        // Time the solve
        double st = omp_get_wtime();
        std::tie(XX, oldXX) = UpdateBelosBackwardEuler(XX);
        double et = omp_get_wtime() - st;

        // Update 'time'
        t += fct::dt_;

        // Get the error in the extensibility
        auto [XC, YC, TC, ext_err] = skelly_fiber::Extricate<ScalarT, VecT>(XX, FS, fct::length_);

        // Write out some information
        spdlog::info("... Runtime:              {}", et);
        spdlog::info("... Extensibility Error:  {}", ext_err.val());

        // Save information
        runtimes.push_back(et);
        ts.push_back(t);
        max_extensibility_error.push_back(ext_err.val());
        auto deflection_real = skelly_chebyshev::RightEvalPoly<ScalarT, VecT>(XC);
        deflection.push_back(deflection_real.val());
        xx.push_back(XX);
    }

    return TimingResult<VecT>(runtimes, ts, max_extensibility_error, deflection, xx);
}

// ****************************************************************************
// Single step Newton solve
// ****************************************************************************
// Directly computes the Jacobian inverse to solve the linearized system

// Single step of a Newton solve by hand, using the inverse of the Jacobian
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

// Timestepping function for the newton version of the solve
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
template <typename VecT>
TimingResult<VecT> run_and_track_singlenewton() {
    // Cute using to get the scalar type of the vector
    using ScalarT = typename VecT::Scalar;
    // Fancy printing in eigen
    Eigen::IOFormat ColumnAsRowFmt(Eigen::StreamPrecision, 0, ",", ",", "", "", "[", "]");

    // Figure out the timesteps we are doing
    int nsteps = int(fct::max_time_ / fct::dt_);

    // Create a fiber solver and initial state
    // NOTE: This is what sets the type(s) for everything...
    auto [FS, XX] = SetupSolverInitialstate<VecT>(fct::N_, fct::length_);
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

    // Vector of results to keep track of...
    std::vector<double> runtimes;
    std::vector<double> ts;
    std::vector<double> max_extensibility_error;
    std::vector<double> deflection;
    std::vector<VecT> xx;

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
        auto [XC, YC, TC, ext_err] = skelly_fiber::Extricate<ScalarT, VecT>(XX, FS, fct::length_);

        // Write out some information
        spdlog::info("... Runtime:              {}", et);
        spdlog::info("... Extensibility Error:  {}", ext_err.val());

        // Save information
        runtimes.push_back(et);
        ts.push_back(t);
        max_extensibility_error.push_back(ext_err.val());
        auto deflection_real = skelly_chebyshev::RightEvalPoly<ScalarT, VecT>(XC);
        deflection.push_back(deflection_real.val());
        xx.push_back(XX);
    }

    return TimingResult<VecT>(runtimes, ts, max_extensibility_error, deflection, xx);
}

// Try to write a solver for the a chebyshev fiber in the sheer deflection flow based on how we solve the full
// system
int main(int argc, char *argv[]) {
    // Assuming we have an argument (first one) that determines what kind of system we are running
    std::string run_type(argv[1]);

    // Set up some using directives to make calling things easier...
    using namespace skelly_fiber;

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

    // Here are all of the configurations that we're going to run
    double master_zeta = 1000.0;
    double max_time = 3.0 / master_zeta;
    std::vector<double> dts_vec{4.0, 8.0, 16.0, 32.0};
    std::vector<double> dts;
    for (auto idt : dts_vec) {
        dts.push_back(1.0 / master_zeta / idt);
    }
    std::vector<int> NS{20, 40, 80};
    double tolerance = 1e-10;

    // Create some result framework to save things
    // ["runtype"][N][dt]
    std::unordered_map<std::string,
                       std::unordered_map<int, std::unordered_map<double, TimingResult<autodiff::VectorXreal>>>>
        all_results;

    // Wrap everything in a try/catch block to see if something has gone wrong
    try {
        // Setup our runs in two for loops and keep track of the results
        for (auto N : NS) {
            for (auto dt : dts) {
                spdlog::info("----------------------------------------------------------------");
                spdlog::info("Working on N = {}, dt = {}", N, dt);
                spdlog::info("Run type: {}", run_type);

                // Set the parameter struct/namespace for this run
                fct::N_ = N;
                fct::length_ = 1.0;
                fct::zeta_ = master_zeta;
                fct::dt_ = dt;
                fct::max_time_ = max_time;
                fct::tolerance_ = tolerance;

                // Call the run function
                if (run_type == "jnewton") {
                    all_results["jnewton"][N][dt] = run_and_track_singlenewton<autodiff::VectorXreal>();
                } else if (run_type == "belos") {
                    all_results["belos"][N][dt] = run_and_track_belos<autodiff::VectorXreal>();
                } else if (run_type == "all") {
                    // Catcher for everything!
                    all_results["jnewton"][N][dt] = run_and_track_singlenewton<autodiff::VectorXreal>();
                    all_results["belos"][N][dt] = run_and_track_belos<autodiff::VectorXreal>();
                } else {
                    throw std::runtime_error("Incorrect run type");
                }
            }
        }

        // Print out something at the end?
        Eigen::IOFormat ColumnAsRowFmt(Eigen::StreamPrecision, 0, ",", ",", "", "", "[", "]");
        spdlog::info("");
        spdlog::info("");
        spdlog::info("-------- Results --------");
        std::vector<std::string> solver_types;
        if (run_type != "all") {
            solver_types.push_back(run_type);
        } else {
            solver_types.push_back("jnewton");
            solver_types.push_back("belos");
        }
        for (auto stype : solver_types) {
            for (auto N : NS) {
                for (auto dt : dts) {
                    auto mresult = all_results[stype][N][dt];
                    spdlog::info("--------------------------------");
                    spdlog::info("Run type = {}, N = {}, dt = {}", stype, N, dt);

                    // Eigen::VectorXd has a prettier line print at the moment...
                    Eigen::VectorXd mdeflection =
                        Eigen::VectorXd::Map(&mresult.deflection_[0], mresult.deflection_.size());
                    Eigen::VectorXd mruntime = Eigen::VectorXd::Map(&mresult.runtimes_[0], mresult.runtimes_.size());

                    std::ostringstream deflection_stream;
                    deflection_stream << "deflection = " << mdeflection.format(ColumnAsRowFmt);
                    spdlog::info(deflection_stream.str());

                    std::ostringstream runtime_stream;
                    runtime_stream << "runtime = " << mruntime.format(ColumnAsRowFmt);
                    spdlog::info(runtime_stream.str());
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
