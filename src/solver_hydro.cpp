#include <solver_hydro.hpp>
#include <system.hpp>

#include <Teuchos_ParameterList.hpp>

#include <BelosBlockGmresSolMgr.hpp>
#include <BelosLinearProblem.hpp>
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
    const auto [fiber_sol_size, shell_sol_size, body_sol_size] = System::get_instance().get_local_solution_sizes();
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
    VectorMap RHS_body(RHS_->getDataNonConst(0).getRawPtr() + fib_sol_size + shell_sol_size, body_sol_size);

    // Initialize GMRES RHS vector
    RHS_fib = System::get_fiber_container().get_RHS();
    RHS_shell = System::get_shell().get_RHS();
    RHS_body = System::get_body_container().get_RHS();
}

template <>
void Solver<P_inv_hydro, A_fiber_hydro>::solve() {
    const int rank = comm_->getRank();
    Belos::LinearProblem<ST, MV, OP> problem(matvec_, X_, RHS_);
    problem.setRightPrec(preconditioner_);
    bool set = problem.setProblem();

    Teuchos::ParameterList belosList;
    belosList.set("Block Size", 1);                                         // Blocksize to be used by iterative solver
    belosList.set("Maximum Iterations", 100);                               // Maximum number of iterations allowed
    belosList.set("Convergence Tolerance", System::get_params().gmres_tol); // Relative convergence tolerance requested
    belosList.set("Orthogonalization", "DGKS");                             // Orthogonalization type

    Belos::BlockGmresSolMgr<ST, MV, OP> solver(rcpFromRef(problem), rcpFromRef(belosList));

    double st = omp_get_wtime();
    Belos::ReturnType ret = solver.solve();
    if (ret == Belos::Converged) {
        if (rank == 0) {
            std::cout << "Solver converged\n";
            std::cout << solver.getNumIters() << " " << omp_get_wtime() - st << " " << solver.achievedTol()
                      << std::endl;
        }
    } else if (rank == 0) {
        std::cout << "Solver failed to converge\n";
    }
}
