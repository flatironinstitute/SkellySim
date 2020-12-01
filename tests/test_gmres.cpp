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

using namespace Teuchos;
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
    P_inv_hydro(const Teuchos::RCP<const Teuchos::Comm<int>> comm, const FiberContainer &fc, const Periphery &shell)
        : fc_(fc), shell_(shell) {
        using Teuchos::rcp;
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
        opMap_ =
            rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), local_size, indexBase, comm));
    };
    //
    // These functions are required since we inherit from Tpetra::Operator
    //
    // Destructor
    virtual ~P_inv_hydro() {}
    // Get the domain Map of this Operator subclass.
    Teuchos::RCP<const map_type> getDomainMap() const { return opMap_; }
    // Get the range Map of this Operator subclass.
    Teuchos::RCP<const map_type> getRangeMap() const { return opMap_; }
    // Compute Y := alpha Op X + beta Y.
    //
    // We ignore the cases alpha != 1 and beta != 0 for simplicity.
    void apply(const MV &X, MV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
               scalar_type alpha = Teuchos::ScalarTraits<scalar_type>::one(),
               scalar_type beta = Teuchos::ScalarTraits<scalar_type>::zero()) const {
        RCP<const Teuchos::Comm<int>> comm = opMap_->getComm();
        const int rank = comm->getRank();
        const int size = comm->getSize();
        if (rank == 0) {
            cout << "P_inv_hydro::apply" << endl;
        }

        const size_t numVecs = X.getNumVectors();
        const local_ordinal_type nlocRows = static_cast<local_ordinal_type>(X.getLocalLength());
        // TEUCHOS_TEST_FOR_EXCEPTION(fc_.fibers.size() == nlocRows, std::logic_error, "")
        const size_t nglobCols = opMap_->getGlobalNumElements();
        const size_t baseRow = opMap_->getMinGlobalIndex();
        const int n_fib_pts = fc_.get_total_fib_points() * 4;
        const int n_shell_pts = shell_.M_inv_.rows();
        for (size_t c = 0; c < numVecs; ++c) {
            using Eigen::Map;
            using Eigen::VectorXd;

            // Get a view of the desired column
            local_ordinal_type offset = 0;
            for (auto &fib : fc_.fibers) {
                Map<const VectorXd> XView(X.getData(c).getRawPtr() + offset, fib.num_points_ * 4);
                auto res = fib.A_LU_.solve(XView);

                for (local_ordinal_type i = 0; i < fib.num_points_ * 4; ++i) {
                    Y.replaceLocalValue(i + offset, c, res(i));
                }

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
    Teuchos::RCP<const map_type> opMap_;
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
    A_fiber_hydro(const Teuchos::RCP<const Teuchos::Comm<int>> comm, const FiberContainer &fc, const Periphery &shell,
                  const double eta)
        : fc_(fc), shell_(shell), eta_(eta) {
        using Teuchos::rcp;
        TEUCHOS_TEST_FOR_EXCEPTION(comm.is_null(), std::invalid_argument,
                                   "A_fiber_hydro constructor: The input Comm object must be nonnull.");
        if (comm->getRank() == 0) {
            cout << "A_fiber_hydro constructor" << endl;
        }

        const global_ordinal_type indexBase = 0;
        const int nfib_pts_local = fc_.get_total_fib_points() * 4;
        const int n_shell_pts_local = shell_.M_inv_.rows();
        std::cout << "initializing a_fiber " << comm->getRank() << " " << nfib_pts_local << " " << n_shell_pts_local
                  << std::endl;
        const int local_size = nfib_pts_local + n_shell_pts_local;

        // Construct a map for our block row distribution
        opMap_ =
            rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), local_size, indexBase, comm));
    };
    //
    // These functions are required since we inherit from Tpetra::Operator
    //
    // Destructor
    virtual ~A_fiber_hydro() {}
    // Get the domain Map of this Operator subclass.
    Teuchos::RCP<const map_type> getDomainMap() const { return opMap_; }
    // Get the range Map of this Operator subclass.
    Teuchos::RCP<const map_type> getRangeMap() const { return opMap_; }

    //     | -0.5*I + T   -K   {G,R}Cbf + Gf         | |w*mu|   |   - G*F - R*L|
    //     |     -K^T      I        0                | | U  | = |      0       |
    //     |    -QoT      Cfb    A_ff - Qo{G,R} Cbf  | | Xf |   |     RHSf     |
    void apply(const MV &X, MV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
               scalar_type alpha = Teuchos::ScalarTraits<scalar_type>::one(),
               scalar_type beta = Teuchos::ScalarTraits<scalar_type>::zero()) const {
        RCP<const Teuchos::Comm<int>> comm = opMap_->getComm();
        const int rank = comm->getRank();
        const int size = comm->getSize();
        if (rank == 0) {
            cout << "A_fiber_hydro::apply" << endl;
        }

        const int nfib_pts_local = 4 * fc_.get_total_fib_points();
        for (size_t c = 0; c < X.getNumVectors(); ++c) {
            local_ordinal_type offset = 0;
            using Eigen::Map;
            using Eigen::MatrixXd;
            using Eigen::VectorXd;

            // Get views and temporary arrays
            double *res_ptr = Y.getDataNonConst(c).getRawPtr();
            const double *x_ptr = X.getData(c).getRawPtr();
            Map<const VectorXd> x_fib_local(x_ptr, nfib_pts_local);
            Map<const VectorXd> x_shell_local(x_ptr + offset, shell_.node_counts_[rank]);
            Map<VectorXd> res_fib(res_ptr, nfib_pts_local);
            MatrixXd r_fib = fc_.get_r_vectors();

            // calculate fiber-fiber velocity
            MatrixXd fw = fc_.apply_fiber_force(x_fib_local);
            MatrixXd v_fib = fc_.flow(fw, eta_);

            Map<VectorXd> res_view_shell(res_ptr + offset, shell_.node_counts_[rank]);
            VectorXd x_shell_global(3 * shell_.n_nodes_global_);

            // TODO: encapsulate all-to-all, or handle it via overlapping Tpetra::Map
            offset += nfib_pts_local;

            MPI_Allgatherv(x_ptr + offset, shell_.node_counts_[rank], MPI_DOUBLE, x_shell_global.data(),
                           shell_.node_counts_.data(), shell_.node_displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);

            res_view_shell = shell_.stresslet_plus_complementary_ * x_shell_global;

            Eigen::MatrixXd vshell2fib = shell_.flow(r_fib, x_shell_local, eta_);

            v_fib += vshell2fib;
            res_fib = fc_.matvec(x_fib_local, v_fib);
        }
    }

  private:
    Teuchos::RCP<const map_type> opMap_;
    const FiberContainer &fc_;
    const Periphery &shell_;
    const double eta_;
};

Eigen::VectorXd load_vec(cnpy::npz_t &npz, const char *var) {
    return Eigen::Map<Eigen::VectorXd>(npz[var].data<double>(), npz[var].shape[0]);
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

        const int rank = comm->getRank();
        const int size = comm->getSize();

        std::ifstream ifs("2K_MTs_onCortex_R5_L1.fibers");
        std::string token;
        getline(ifs, token);
        const int nfibs_tot = atoi(token.c_str());
        const int nfibs_extra = nfibs_tot % size;
        const int n_pts = 32;
        const int n_time = 10;
        const double eta = 10.0;
        const double bending_rigidity = 0.1;
        const double length = 1.0;
        double dt = 1E-4;
        std::vector<int> displs(size + 1);
        for (int i = 1; i < size + 1; ++i) {
            displs[i] = displs[i - 1] + nfibs_tot / size;
            if (i <= nfibs_extra)
                displs[i]++;
        }

        assert(nfibs_tot % size == 0);
        const int n_fibs_local = displs[rank + 1] - displs[rank];
        Periphery shell("test_periphery.npz");
        FiberContainer fibs(n_fibs_local, n_pts, bending_rigidity, eta);

        if (rank == 0)
            cout << "Reading in " << nfibs_tot << " fibers.\n";

        // FIXME: Fiber import ludicrously slow to compile
        for (int ifib = 0; ifib < nfibs_tot; ++ifib) {
            const int ifib_low = displs[rank];
            const int ifib_high = displs[rank + 1];
            std::string line;
            getline(ifs, line);
            std::stringstream linestream(line);

            getline(linestream, token, ' ');
            int npts = atoi(token.c_str());

            getline(linestream, token, ' ');
            double E = bending_rigidity; // atof(token.c_str());

            getline(linestream, token, ' ');
            double L = atof(token.c_str());

            assert(npts == n_pts);
            assert(E == bending_rigidity);
            assert(L == length);

            Eigen::MatrixXd x(3, n_pts);
            for (int ipt = 0; ipt < npts; ++ipt) {
                getline(ifs, line);
                std::stringstream linestream(line);

                if (ifib >= ifib_low && ifib < ifib_high) {
                    for (int i = 0; i < 3; ++i) {
                        getline(linestream, token, ' ');

                        x(i, ipt) = atof(token.c_str());
                    }
                }
            }

            if (ifib >= ifib_low && ifib < ifib_high) {
                cout << "Fiber " << ifib << ": " << npts << " " << E << " " << L << endl;
                Eigen::MatrixXd v_on_fiber;
                Eigen::MatrixXd f_on_fiber;
                auto &fib = fibs.fibers[ifib - ifib_low];

                fib.x_ = x;
                fib.length_ = length;
                fib.update_derivatives();
                fib.update_stokeslet(eta);
                fib.form_linear_operator(dt, eta);
                fib.compute_RHS(dt, v_on_fiber, f_on_fiber);
                fib.apply_bc_rectangular(dt, v_on_fiber, f_on_fiber);
                fib.build_preconditioner();
                fib.form_force_operator();
            }
        }

        RCP<A_fiber_hydro> A_sim = rcp(new A_fiber_hydro(comm, fibs, shell, eta));
        RCP<P_inv_hydro> preconditioner = rcp(new P_inv_hydro(comm, fibs, shell));

        RCP<const Tpetra::Map<>> map = A_sim->getDomainMap();
        typedef Tpetra::Vector<A_fiber_hydro::scalar_type, A_fiber_hydro::local_ordinal_type,
                               A_fiber_hydro::global_ordinal_type, A_fiber_hydro::node_type>
            vec_type;

        // Create initial vectors
        RCP<vec_type> X, RHS;
        X = rcp(new vec_type(map));
        RHS = rcp(new vec_type(map));
        X->putScalar(0.0);

        { // Initialize FMM for reliable benchmarks
            double tmp = fibs.fibers[0].x_(0, 0);
            fibs.fibers[0].x_(0, 0) = 1.0;
            A_sim->apply(*X, *RHS);
            fibs.fibers[0].x_(0, 0) = tmp;
        }

        RHS->putScalar(0.0);
        { // Initialize RHS
            int offset = 0;
            for (auto &fib : fibs.fibers) {
                std::memcpy(RHS->getDataNonConst(0).getRawPtr() + offset, fib.RHS_.data(), fib.RHS_.size());
                offset += fib.RHS_.size();
            }

            // Initialize RHS for shell
            // Just the velocity, which should be zero on first pass
            // So.. do nothing
            offset += shell.M_inv_.rows();
            // std::cout << offset << " " << map->getLocalMap().getNodeNumElements() << std::endl;
        }

        Belos::LinearProblem<ST, MV, OP> problem(A_sim, X, RHS);
        if (rank == 0)
            std::cout << "Initialized linear problem\n";

        int blocksize = 1;         // blocksize used by solver
        std::string ortho("DGKS"); // orthogonalization type
        double tol = 1.0E-7;       // relative residual tolerance
        int prec_flag = true;

        CommandLineProcessor cmdp(false, true);
        cmdp.setOption("prec-flag", &prec_flag, "Enable preconditioner.");
        cmdp.setOption("tol", &tol, "Relative residual tolerance used by Gmres solver.");
        cmdp.setOption("block-size", &blocksize, "Block size to be used by the Gmres solver.");
        cmdp.setOption("ortho-type", &ortho, "Orthogonalization type, either DGKS, ICGS or IMGS (or TSQR if enabled)");
        if (cmdp.parse(argc, argv) != CommandLineProcessor::PARSE_SUCCESSFUL) {
            return -1;
        }

        // TODO: right preconditioner is correct?
        if (prec_flag) {
            problem.setRightPrec(preconditioner);
            if (rank == 0)
                std::cout << "Set preconditioner\n";
        }

        bool set = problem.setProblem();
        if (set == false) {
            cout << endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << endl;
            return -1;
        }
        if (rank == 0)
            std::cout << "Set Belos problem\n";

        ParameterList belosList;
        belosList.set("Block Size", blocksize);      // Blocksize to be used by iterative solver
        belosList.set("Maximum Iterations", 100);    // Maximum number of iterations allowed
        belosList.set("Convergence Tolerance", tol); // Relative convergence tolerance requested
        belosList.set("Orthogonalization", ortho);   // Orthogonalization type

        Belos::BlockGmresSolMgr<ST, MV, OP> solver(rcpFromRef(problem), rcpFromRef(belosList));
        if (rank == 0)
            std::cout << "Initialized GMRES solver\n";

        // RCP<Teuchos::FancyOStream> fos = rcp(new Teuchos::FancyOStream(rcpFromRef(std::cout)));
        // X->describe(*fos, Teuchos::VERB_EXTREME);
        // RHS->describe(*fos, Teuchos::VERB_EXTREME);

        double st = omp_get_wtime();
        Belos::ReturnType ret = solver.solve();
        if (rank == 0)
            cout << solver.getNumIters() << " " << omp_get_wtime() - st << endl;

        success = true;
        if (rank == 0)
            cout << "Test passed\n";
    }
    // TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

    return (success ? EXIT_SUCCESS : EXIT_FAILURE);
}
