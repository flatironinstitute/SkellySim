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
    P_inv_hydro(const RCP<const Comm<int>> comm, const FiberContainer &fc, const Periphery &shell)
        : fc_(fc), shell_(shell) {
        TEUCHOS_TEST_FOR_EXCEPTION(comm.is_null(), std::invalid_argument,
                                   "P_inv_hydro constructor: The input Comm object must be nonnull.");
        if (comm->getRank() == 0) {
            cout << "P_inv_hydro constructor" << endl;
        }

        const global_ordinal_type indexBase = 0;
        const int nfib_pts_local = fc_.get_total_fib_points() * 4;
        const int n_shell_rows_local = shell_.M_inv_.rows();
        const int local_size = nfib_pts_local + n_shell_rows_local;

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
            using Eigen::Map;

            // Get a view of the desired column
            local_ordinal_type offset = 0;
            for (auto &fib : fc_.fibers) {
                Map<const VectorXd> XView(X.getData(c).getRawPtr() + offset, fib.num_points_ * 4);
                Map<VectorXd> res_fib(Y.getDataNonConst(c).getRawPtr() + offset, fib.num_points_ * 4);
                res_fib = fib.A_LU_.solve(XView);

                offset += fib.num_points_ * 4;
            }

            Map<VectorXd> res_view_shell(Y.getDataNonConst(c).getRawPtr() + offset, shell_.node_counts_[rank]);
            VectorXd x_shell(3 * shell_.n_nodes_global_);

            MPI_Allgatherv(X.getData(c).getRawPtr() + offset, shell_.node_counts_[rank], MPI_DOUBLE, x_shell.data(),
                           shell_.node_counts_.data(), shell_.node_displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);

            res_view_shell = shell_.M_inv_ * x_shell;
        }
    }

  private:
    RCP<const map_type> opMap_;
    const FiberContainer &fc_;
    const Periphery &shell_;
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
    A_fiber_hydro(const RCP<const Comm<int>> comm, const FiberContainer &fc, const Periphery &shell, const double eta)
        : fc_(fc), shell_(shell), eta_(eta) {
        TEUCHOS_TEST_FOR_EXCEPTION(comm.is_null(), std::invalid_argument,
                                   "A_fiber_hydro constructor: The input Comm object must be nonnull.");
        if (comm->getRank() == 0) {
            cout << "A_fiber_hydro constructor" << endl;
        }

        const global_ordinal_type indexBase = 0;
        const int nfib_pts_local = fc_.get_total_fib_points() * 4;
        const int n_shell_pts_local = shell_.M_inv_.rows();
        const int local_size = nfib_pts_local + n_shell_pts_local;

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

        const int n_fib_pts_local = 4 * fc_.get_total_fib_points();
        const int n_shell_pts_local = shell_.node_counts_[rank];
        for (size_t c = 0; c < X.getNumVectors(); ++c) {
            using Eigen::Map;

            // Get views and temporary arrays
            double *res_ptr = Y.getDataNonConst(c).getRawPtr();
            const double *x_ptr = X.getData(c).getRawPtr();
            Map<const VectorXd> x_fib_local(x_ptr, n_fib_pts_local);
            Map<const VectorXd> x_shell_local(x_ptr + n_fib_pts_local, n_shell_pts_local);
            Map<VectorXd> res_fib(res_ptr, n_fib_pts_local);
            Map<VectorXd> res_shell(res_ptr + n_fib_pts_local, n_shell_pts_local);
            MatrixXd r_fib = fc_.get_r_vectors();
            VectorXd x_shell_global(3 * shell_.n_nodes_global_);
            MPI_Allgatherv(x_shell_local.data(), n_shell_pts_local, MPI_DOUBLE, x_shell_global.data(),
                           shell_.node_counts_.data(), shell_.node_displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);

            // calculate fiber-fiber velocity
            MatrixXd fw = fc_.apply_fiber_force(x_fib_local);
            MatrixXd v_all = fc_.flow(fw, shell_.node_pos_, eta_);
            MatrixXd vshell2fib = shell_.flow(r_fib, x_shell_local, eta_);
            v_all.block(0, 0, 3, r_fib.cols()) += vshell2fib;

            res_fib = fc_.matvec(x_fib_local, v_all.block(0, 0, 3, r_fib.cols()));
            res_shell = shell_.stresslet_plus_complementary_ * x_shell_global;
            res_shell += Map<VectorXd>(v_all.data() + 3 * r_fib.cols(), n_shell_pts_local);
        }
    }

  private:
    RCP<const map_type> opMap_;
    const FiberContainer &fc_;
    const Periphery &shell_;
    const double eta_;
};

VectorXd load_vec(cnpy::npz_t &npz, const char *var) {
    return Eigen::Map<VectorXd>(npz[var].data<double>(), npz[var].shape[0]);
}

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

        System system(config_file);

        Params &params = system.params_;
        Periphery &shell = system.shell_;
        FiberContainer &fc = system.fc_;
        const double eta = params.eta;
        const double dt = params.dt;
        const double tol = params.gmres_tol;

        for (auto &fib : fc.fibers) {
            fib.update_derivatives();
            fib.update_stokeslet(eta);
            fib.update_linear_operator(dt, eta);
        }

        {
            MatrixXd r_trg_external = shell.node_pos_;
            MatrixXd f_on_fibers = fc.generate_constant_force();
            MatrixXd v_fib2all = fc.flow(f_on_fibers, r_trg_external, eta);
            size_t offset = 0;
            for (auto &fib : fc.fibers) {
                fib.update_RHS(dt, v_fib2all.block(0, offset, 3, fib.num_points_),
                               f_on_fibers.block(0, offset, 3, fib.num_points_));
                fib.apply_bc_rectangular(dt, v_fib2all.block(0, offset, 3, fib.num_points_),
                                         f_on_fibers.block(0, offset, 3, fib.num_points_));
                fib.update_preconditioner();
                fib.update_force_operator();
                offset += fib.num_points_;
            }

            shell.update_RHS(v_fib2all.block(0, offset, 3, r_trg_external.cols()));
        }

        RCP<A_fiber_hydro> A_sim = rcp(new A_fiber_hydro(comm, fc, shell, eta));
        RCP<P_inv_hydro> preconditioner = rcp(new P_inv_hydro(comm, fc, shell));

        RCP<const Tpetra::Map<>> map = A_sim->getDomainMap();
        typedef Tpetra::Vector<A_fiber_hydro::scalar_type, A_fiber_hydro::local_ordinal_type,
                               A_fiber_hydro::global_ordinal_type, A_fiber_hydro::node_type>
            vec_type;

        // Create initial vectors
        RCP<vec_type> X, RHS;
        X = rcp(new vec_type(map));
        RHS = rcp(new vec_type(map));
        X->putScalar(1.0);

        A_sim->apply(*X, *RHS);

        cnpy::npy_save("Y.npy", RHS->getData().getRawPtr(), {RHS->getLocalLength()});
        X->putScalar(0.0);
        { // Initialize FMM for reliable benchmarks
            double tmp = fc.fibers[0].x_(0, 0);
            fc.fibers[0].x_(0, 0) = 1.0;
            A_sim->apply(*X, *RHS);
            fc.fibers[0].x_(0, 0) = tmp;
        }

        RHS->putScalar(0.0);
        { // Initialize RHS
            int offset = 0;
            for (auto &fib : fc.fibers) {
                Eigen::Map<Eigen::VectorXd>(RHS->getDataNonConst(0).getRawPtr() + offset, fib.RHS_.size()) = fib.RHS_;
                offset += fib.RHS_.size();
            }

            // Initialize RHS for shell
            // Just the velocity, which should be zero on first pass
            // So.. do nothing
            Eigen::Map<Eigen::VectorXd>(RHS->getDataNonConst(0).getRawPtr() + offset, shell.RHS_.size()) = shell.RHS_;
            offset += shell.RHS_.size();
        }
        cnpy::npy_save("RHS.npy", RHS->getData().getRawPtr(), {RHS->getLocalLength()});

        Belos::LinearProblem<ST, MV, OP> problem(A_sim, X, RHS);
        if (rank == 0)
            cout << "Initialized linear problem\n";

        // TODO: right preconditioner is correct?
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
