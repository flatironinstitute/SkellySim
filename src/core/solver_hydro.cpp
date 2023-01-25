#include <params.hpp>
#include <solver_hydro.hpp>
#include <system.hpp>
#include <utils.hpp>

#include <Teuchos_ParameterList.hpp>

#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosTpetraAdapter.hpp>

P_inv_hydro::P_inv_hydro(const Teuchos::RCP<const Teuchos::Comm<int>> comm) : comm_(comm), rank_(comm->getRank()) {
    TEUCHOS_TEST_FOR_EXCEPTION(comm.is_null(), std::invalid_argument,
                               "P_inv_hydro constructor: The input Teuchos::Comm object must be nonnull.");
    const global_ordinal_type indexBase = 0;
    const auto [fiber_sol_size, shell_sol_size, body_sol_size] = System::get_local_solution_sizes();
    const int local_size = fiber_sol_size + shell_sol_size + body_sol_size;

    // Construct a map for our block row distribution
    opMap_ = rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), local_size, indexBase, comm));
}

void P_inv_hydro::apply(const MV &X, MV &Y, Teuchos::ETransp mode, scalar_type alpha, scalar_type beta) const {
    for (size_t c = 0; c < X.getNumVectors(); ++c) {
        CVectorMap x_local(X.getData(c).getRawPtr(), X.getLocalLength());
        VectorMap res(Y.getDataNonConst(c).getRawPtr(), Y.getLocalLength());
        res = System::apply_preconditioner(x_local);
    }
}

A_fiber_hydro::A_fiber_hydro(const Teuchos::RCP<const Teuchos::Comm<int>> comm) : comm_(comm), rank_(comm->getRank()) {
    TEUCHOS_TEST_FOR_EXCEPTION(comm.is_null(), std::invalid_argument,
                               "A_fiber_hydro constructor: The input Comm object must be nonnull.");
    const global_ordinal_type indexBase = 0;
    const auto [fiber_sol_size, shell_sol_size, body_sol_size] = System::get_local_solution_sizes();
    const int local_size = fiber_sol_size + shell_sol_size + body_sol_size;

    // Construct a map for our block row distribution
    opMap_ = rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), local_size, indexBase, comm));
};

void A_fiber_hydro::apply(const MV &X, MV &Y, Teuchos::ETransp mode, scalar_type alpha, scalar_type beta) const {
    for (size_t c = 0; c < X.getNumVectors(); ++c) {
        CVectorMap x_local(X.getData(c).getRawPtr(), X.getLocalLength());
        VectorMap res(Y.getDataNonConst(c).getRawPtr(), Y.getLocalLength());
        res = System::apply_matvec(x_local);
    }
}

template <>
void Solver<P_inv_hydro, A_fiber_hydro>::set_RHS() {
    const auto [fib_sol_size, shell_sol_size, body_sol_size] = System::get_local_solution_sizes();
    VectorMap RHS_fib(RHS_->getDataNonConst(0).getRawPtr(), fib_sol_size);
    VectorMap RHS_shell(RHS_->getDataNonConst(0).getRawPtr() + fib_sol_size, shell_sol_size);

    // Initialize GMRES RHS vector
    RHS_fib = System::get_fiber_RHS();
    RHS_shell = System::get_shell_RHS();
}

template <>
bool Solver<P_inv_hydro, A_fiber_hydro>::solve() {
    Belos::LinearProblem<ST, MV, OP> problem(matvec_, X_, RHS_);
    problem.setRightPrec(preconditioner_);
    problem.setProblem();

    Teuchos::ParameterList belosList;
    // allowed
    belosList.set("Convergence Tolerance", System::get_params()->gmres_tol); // Relative convergence tolerance requested
    belosList.set("Orthogonalization", "ICGS");                              // Orthogonalization type
    belosList.set("Verbosity",
                  Belos::MsgType::IterationDetails + Belos::MsgType::FinalSummary + Belos::MsgType::StatusTestDetails);
    belosList.set("Output Frequency", 1);
    belosList.set("Output Style", Belos::OutputType::General);

    Belos::PseudoBlockGmresSolMgr<ST, MV, OP> solver(rcpFromRef(problem), rcpFromRef(belosList));
    utils::LoggerRedirect redirect(std::cout);

    double st = omp_get_wtime();
    Belos::ReturnType ret = solver.solve();
    redirect.flush(spdlog::level::trace, "Belos");

    if (ret == Belos::Converged) {
        spdlog::info("Solver converged with parameters: iters {}, time {}, achieved tolerance {}", solver.getNumIters(),
                     omp_get_wtime() - st, solver.achievedTol());
    } else {
        spdlog::info("Solver failed to converge with parameters: iters {}, time {}, achieved tolerance {}",
                     solver.getNumIters(), omp_get_wtime() - st, solver.achievedTol());
        spdlog::info("loss of accuracy: {}", solver.isLOADetected());
    }

    return ret == Belos::Converged;
}
