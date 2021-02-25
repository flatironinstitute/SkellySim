#include <skelly_sim.hpp>

#include <BelosBlockGmresSolMgr.hpp>
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardCatchMacros.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>

#include <cnpy.hpp>
#include <fiber.hpp>
#include <periphery.hpp>
#include <system.hpp>
#include <utils.hpp>

using namespace Teuchos;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

class P_inv_hydro : public Tpetra::Operator<> {
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
    // Constructor
    //
    // n: Global number of rows and columns in the operator.
    // comm: The communicator over which to distribute those rows and columns.
    P_inv_hydro(const RCP<const Comm<int>> comm)
        : fc_(System::get_fiber_container()), shell_(System::get_shell()), bc_(System::get_body_container()) {
        TEUCHOS_TEST_FOR_EXCEPTION(comm.is_null(), std::invalid_argument,
                                   "P_inv_hydro constructor: The input Comm object must be nonnull.");
        if (comm->getRank() == 0) {
            cout << "P_inv_hydro constructor" << endl;
        }

        const global_ordinal_type indexBase = 0;
        const int fiber_sol_size = fc_.get_local_solution_size();
        const int shell_sol_size = shell_.get_local_solution_size();
        const int body_sol_size = bc_.get_local_solution_size();
        const int local_size = fiber_sol_size + shell_sol_size + body_sol_size;

        // Construct a map for our block row distribution
        opMap_ = rcp(new map_type(OrdinalTraits<Tpetra::global_size_t>::invalid(), local_size, indexBase, comm));
    };
    //
    // These functions are required since we inherit from Tpetra::Operator
    //
    // Destructor
    virtual ~P_inv_hydro() {}
    // Get the domain Map of this Operator subclass.
    RCP<const map_type> getDomainMap() const { return opMap_; }
    // Get the range Map of this Operator subclass.
    RCP<const map_type> getRangeMap() const { return opMap_; }
    // Compute Y := alpha Op X + beta Y.
    //
    // We ignore the cases alpha != 1 and beta != 0 for simplicity.
    void apply(const MV &X, MV &Y, ETransp mode = NO_TRANS, scalar_type alpha = ScalarTraits<scalar_type>::one(),
               scalar_type beta = ScalarTraits<scalar_type>::zero()) const {
        RCP<const Comm<int>> comm = opMap_->getComm();
        const int rank = comm->getRank();

        if (rank == 0) {
            cout << "P_inv_hydro::apply" << endl;
        }

        for (size_t c = 0; c < X.getNumVectors(); ++c) {
            CVectorMap x_local(X.getData(c).getRawPtr(), X.getLocalLength());
            VectorMap res(Y.getDataNonConst(c).getRawPtr(), Y.getLocalLength());
            res = System::apply_preconditioner(x_local);
        }
    }

  private:
    RCP<const map_type> opMap_;
    const FiberContainer &fc_;
    const Periphery &shell_;
    const BodyContainer &bc_;
};

class A_fiber_hydro : public Tpetra::Operator<> {
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
    // Constructor
    //
    // n: Global number of rows and columns in the operator.
    // comm: The communicator over which to distribute those rows and columns.
    A_fiber_hydro(const RCP<const Comm<int>> comm, const double eta)
        : fc_(System::get_fiber_container()), shell_(System::get_shell()), bc_(System::get_body_container()),
          eta_(eta) {
        TEUCHOS_TEST_FOR_EXCEPTION(comm.is_null(), std::invalid_argument,
                                   "A_fiber_hydro constructor: The input Comm object must be nonnull.");
        if (comm->getRank() == 0) {
            cout << "A_fiber_hydro constructor" << endl;
        }

        const global_ordinal_type indexBase = 0;
        const int fiber_sol_size = fc_.get_local_solution_size();
        const int shell_sol_size = shell_.get_local_solution_size();
        const int body_sol_size = bc_.get_local_solution_size();
        const int local_size = fiber_sol_size + shell_sol_size + body_sol_size;

        // Construct a map for our block row distribution
        opMap_ = rcp(new map_type(OrdinalTraits<Tpetra::global_size_t>::invalid(), local_size, indexBase, comm));
    };
    //
    // These functions are required since we inherit from Tpetra::Operator
    //
    // Destructor
    virtual ~A_fiber_hydro() {}
    // Get the domain Map of this Operator subclass.
    RCP<const map_type> getDomainMap() const { return opMap_; }
    // Get the range Map of this Operator subclass.
    RCP<const map_type> getRangeMap() const { return opMap_; }

    //     | -0.5*I + T   -K   {G,R}Cbf + Gf         | |w*mu|   |   - G*F - R*L|
    //     |     -K^T      I        0                | | U  | = |      0       |
    //     |    -QoT      Cfb    A_ff - Qo{G,R} Cbf  | | Xf |   |     RHSf     |
    void apply(const MV &X, MV &Y, ETransp mode = NO_TRANS, scalar_type alpha = ScalarTraits<scalar_type>::one(),
               scalar_type beta = ScalarTraits<scalar_type>::zero()) const {
        RCP<const Comm<int>> comm = opMap_->getComm();
        const int rank = comm->getRank();

        if (rank == 0) {
            cout << "A_fiber_hydro::apply" << endl;
        }

        for (size_t c = 0; c < X.getNumVectors(); ++c) {
            CVectorMap x_local(X.getData(c).getRawPtr(), X.getLocalLength());
            VectorMap res(Y.getDataNonConst(c).getRawPtr(), Y.getLocalLength());
            res = System::apply_matvec(x_local);
        }
    }

  private:
    RCP<const map_type> opMap_;
    const FiberContainer &fc_;
    const Periphery &shell_;
    const BodyContainer &bc_;
    const double eta_;
};

VectorXd load_vec(cnpy::npz_t &npz, const char *var) { return VectorMap(npz[var].data<double>(), npz[var].shape[0]); }

int main(int argc, char *argv[]) {
    typedef double ST;
    typedef Tpetra::Operator<ST> OP;
    typedef Tpetra::MultiVector<ST> MV;

    Tpetra::ScopeGuard tpetraScope(&argc, &argv);

    bool success = false;
    // try
    {
        RCP<const Comm<int>> comm = Tpetra::getDefaultComm();
        RCP<FancyOStream> fos = fancyOStream(rcpFromRef(cout));

        const int rank = comm->getRank();
        const int size = comm->getSize();

        std::string config_file = "test_gmres.toml";
        int blocksize = 1;         // blocksize used by solver
        std::string ortho("DGKS"); // orthogonalization type
        int prec_flag = true;

        CommandLineProcessor cmdp(false, true);
        cmdp.setOption("config-file", &config_file, "TOML input file.");
        cmdp.setOption("prec-flag", &prec_flag, "Enable preconditioner.");
        cmdp.setOption("block-size", &blocksize, "Block size to be used by the Gmres solver.");
        cmdp.setOption("ortho-type", &ortho, "Orthogonalization type, either DGKS, ICGS or IMGS (or TSQR if enabled)");
        if (cmdp.parse(argc, argv) != CommandLineProcessor::PARSE_SUCCESSFUL) {
            return -1;
        }

        System::init(config_file);

        Params &params = System::get_params();
        Periphery &shell = System::get_shell();
        FiberContainer &fc = System::get_fiber_container();
        BodyContainer &bc = System::get_body_container();
        const double eta = params.eta;
        const double dt = params.dt;
        const double tol = params.gmres_tol;

        for (auto &fib : fc.fibers) {
            fib.update_derivatives();
            fib.update_stokeslet(eta);
            fib.update_linear_operator(dt, eta);
        }

        {
            MatrixXd r_trg_external(3, shell.get_local_node_count() + bc.get_local_node_count());
            r_trg_external.block(0, 0, 3, shell.get_local_node_count()) = shell.get_local_node_positions();
            r_trg_external.block(0, shell.get_local_node_count(), 3, bc.get_local_node_count()) =
                bc.get_local_node_positions();

            MatrixXd f_on_fibers = fc.generate_constant_force();
            MatrixXd v_fib2all = fc.flow(f_on_fibers, r_trg_external, eta);
            size_t offset = 0;
            for (auto &fib : fc.fibers) {
                fib.update_RHS(dt, v_fib2all.block(0, offset, 3, fib.n_nodes_),
                               f_on_fibers.block(0, offset, 3, fib.n_nodes_));
                fib.apply_bc_rectangular(dt, v_fib2all.block(0, offset, 3, fib.n_nodes_),
                                         f_on_fibers.block(0, offset, 3, fib.n_nodes_));
                fib.update_preconditioner();
                fib.update_force_operator();
                offset += fib.n_nodes_;
            }

            shell.update_RHS(v_fib2all.block(0, offset, 3, shell.get_local_node_count()));
            offset += shell.get_local_node_count();

            // FIXME: Body update_RHS
            bc.update_cache_variables(eta);
            bc.update_RHS(v_fib2all.block(0, offset, 3, bc.get_local_node_count()));
        }

        RCP<A_fiber_hydro> A_sim = rcp(new A_fiber_hydro(comm, eta));
        RCP<P_inv_hydro> preconditioner = rcp(new P_inv_hydro(comm));

        RCP<const Tpetra::Map<>> map = A_sim->getDomainMap();
        typedef Tpetra::Vector<A_fiber_hydro::scalar_type, A_fiber_hydro::local_ordinal_type,
                               A_fiber_hydro::global_ordinal_type, A_fiber_hydro::node_type>
            vec_type;

        // Create initial vectors
        RCP<vec_type> X, RHS, X_guess;
        X = rcp(new vec_type(map));
        RHS = rcp(new vec_type(map));
        X_guess = rcp(new vec_type(map));

        X->putScalar(0.0);
        RHS->putScalar(0.0);
        const int fib_sol_size = fc.get_local_solution_size();
        const int shell_sol_size = shell.get_local_solution_size();
        const int body_sol_size = bc.get_local_solution_size();
        VectorMap RHS_fib(RHS->getDataNonConst(0).getRawPtr(), fib_sol_size);
        VectorMap RHS_shell(RHS->getDataNonConst(0).getRawPtr() + fib_sol_size, shell_sol_size);
        VectorMap RHS_body(RHS->getDataNonConst(0).getRawPtr() + fib_sol_size + shell_sol_size, body_sol_size);

        // Initialize GMRES RHS vector
        RHS_fib = fc.get_RHS();
        RHS_shell = shell.get_RHS();
        RHS_body = bc.get_RHS();
        preconditioner->apply(*RHS, *X_guess);

        // Output application of A_hydro operator on simple input for comparison to python output
        {
            RCP<vec_type> Y = rcp(new vec_type(map));
            VectorMap fib_Y(Y->getDataNonConst(0).getRawPtr(), fib_sol_size);
            VectorMap shell_Y(Y->getDataNonConst(0).getRawPtr() + fib_sol_size, shell_sol_size);
            VectorMap body_Y(Y->getDataNonConst(0).getRawPtr() + fib_sol_size + shell_sol_size, body_sol_size);
            VectorMap X_guess_fib(X_guess->getDataNonConst(0).getRawPtr(), fib_sol_size);
            VectorMap X_guess_shell(X_guess->getDataNonConst(0).getRawPtr() + fib_sol_size, shell_sol_size);
            VectorMap X_guess_body(X_guess->getDataNonConst(0).getRawPtr() + fib_sol_size + shell_sol_size,
                                   body_sol_size);

            X->putScalar(1.0);
            A_sim->apply(*X, *Y);
            X->putScalar(0.0);

            Eigen::VectorXd RHS_fib_global = utils::collect_into_global(RHS_fib);
            Eigen::VectorXd RHS_shell_global = utils::collect_into_global(RHS_shell);
            Eigen::VectorXd fib_Y_global = utils::collect_into_global(fib_Y);
            Eigen::VectorXd shell_Y_global = utils::collect_into_global(shell_Y);
            // Eigen::VectorXd X_guess_global = utils::collect_into_global(X_guess_eigen);

            if (rank == 0) {
                cnpy::npy_save("RHS_fib.npy", RHS_fib_global.data(), {(unsigned long)RHS_fib_global.size()});
                cnpy::npy_save("RHS_shell.npy", RHS_shell_global.data(), {(unsigned long)RHS_shell_global.size()});
                cnpy::npy_save("RHS_body.npy", RHS_body.data(), {(unsigned long)RHS_body.size()});
                cnpy::npy_save("Y_fib.npy", fib_Y_global.data(), {(unsigned long)fib_Y_global.size()});
                cnpy::npy_save("Y_shell.npy", shell_Y_global.data(), {(unsigned long)shell_Y_global.size()});
                cnpy::npy_save("Y_body.npy", body_Y.data(), {(unsigned long)body_Y.size()});
                cnpy::npy_save("X_guess_fib.npy", X_guess_fib.data(), {(unsigned long)X_guess_fib.size()});
                cnpy::npy_save("X_guess_shell.npy", X_guess_shell.data(), {(unsigned long)X_guess_shell.size()});
                cnpy::npy_save("X_guess_body.npy", X_guess_body.data(), {(unsigned long)X_guess_body.size()});
            }
        }

        if (fc.fmm_ != nullptr)
            fc.fmm_->force_setup_tree();
        if (shell.fmm_ != nullptr)
            shell.fmm_->force_setup_tree();
        if (bc.oseen_kernel_ != nullptr)
            bc.oseen_kernel_->force_setup_tree();
        if (bc.stresslet_kernel_ != nullptr)
            bc.stresslet_kernel_->force_setup_tree();

        Belos::LinearProblem<ST, MV, OP> problem(A_sim, X, RHS);
        if (rank == 0)
            cout << "Initialized linear problem\n";

        if (prec_flag) {
            problem.setRightPrec(preconditioner);
            if (rank == 0)
                cout << "Set preconditioner\n";
        }

        bool set = problem.setProblem();
        if (set == false) {
            cout << endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << endl;
            return -1;
        }
        if (rank == 0)
            cout << "Set Belos problem\n";

        ParameterList belosList;
        belosList.set("Block Size", blocksize);      // Blocksize to be used by iterative solver
        belosList.set("Maximum Iterations", 100);    // Maximum number of iterations allowed
        belosList.set("Convergence Tolerance", tol); // Relative convergence tolerance requested
        belosList.set("Orthogonalization", ortho);   // Orthogonalization type

        Belos::BlockGmresSolMgr<ST, MV, OP> solver(rcpFromRef(problem), rcpFromRef(belosList));
        if (rank == 0)
            cout << "Initialized GMRES solver\n";

        double st = omp_get_wtime();
        Belos::ReturnType ret = solver.solve();
        RCP<Tpetra::Map<>> proc_zero_map =
            rcp(new Tpetra::Map<>(X->getGlobalLength(), rank == 0 ? X->getGlobalLength() : 0, 0, comm));
        RCP<Tpetra::Vector<>> res = rcp(new Tpetra::Vector<>(proc_zero_map));
        Tpetra::Export<> exporter(map, proc_zero_map);
        res->doExport(*problem.getLHS(), exporter, Tpetra::INSERT);

        if (rank == 0) {
            if (ret == Belos::Converged)
                cout << "Solver converged\n";
            cout << solver.getNumIters() << " " << omp_get_wtime() - st << " " << solver.achievedTol() << endl;

            // FIXME: calculate residual even when size != 1
            if (size == 1) {
                RCP<vec_type> RHS_sol = rcp(new vec_type(map));
                A_sim->apply(*problem.getLHS(), *RHS_sol);

                double residual = 0.0;
                for (size_t i = 0; i < RHS_sol->getLocalLength(); ++i) {
                    residual += pow(RHS_sol->getData().getRawPtr()[i] - RHS->getData().getRawPtr()[i], 2);
                }
                std::cout << std::sqrt(residual) << std::endl;

                cnpy::npy_save("res.npy", res->getData().getRawPtr(), {res->getLocalLength()});
            }
        }

        success = true;
        if (rank == 0)
            cout << "Test passed\n";
    }
    // TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

    return (success ? EXIT_SUCCESS : EXIT_FAILURE);
}
