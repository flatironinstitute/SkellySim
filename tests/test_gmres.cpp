#include <BelosBlockGmresSolMgr.hpp>
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardCatchMacros.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>

#include "cnpy.hpp"
#include <fiber.hpp>

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
    P_inv_hydro(const global_ordinal_type n, const Teuchos::RCP<const Teuchos::Comm<int>> comm,
                const FiberContainer &fc)
        : fc_(fc) {
        using Teuchos::rcp;
        TEUCHOS_TEST_FOR_EXCEPTION(comm.is_null(), std::invalid_argument,
                                   "P_inv_hydro constructor: The input Comm object must be nonnull.");
        if (comm->getRank() == 0) {
            cout << "P_inv_hydro constructor" << endl;
        }

        const global_ordinal_type indexBase = 0;
        // Construct a map for our block row distribution
        opMap_ = rcp(new map_type(n, indexBase, comm));
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
        const int myRank = comm->getRank();
        if (myRank == 0) {
            cout << "P_inv_hydro::apply" << endl;
        }

        const size_t numVecs = X.getNumVectors();
        const local_ordinal_type nlocRows = static_cast<local_ordinal_type>(X.getLocalLength());
        // TEUCHOS_TEST_FOR_EXCEPTION(fc_.fibers.size() == nlocRows, std::logic_error, "")
        const size_t nglobCols = opMap_->getGlobalNumElements();
        const size_t baseRow = opMap_->getMinGlobalIndex();
        for (size_t c = 0; c < numVecs; ++c) {
            // Get a view of the desired column
            local_ordinal_type offset = 0;
            for (auto &fib : fc_.fibers) {
                Eigen::Map<const Eigen::VectorXd> XView(X.getData(c).getRawPtr() + offset, fib.num_points_ * 4);
                auto res = fib.A_LU_.solve(XView);

                for (local_ordinal_type i = 0; i < fib.num_points_ * 4; ++i) {
                    Y.replaceLocalValue(i + offset, c, res(i));
                }

                offset += fib.num_points_ * 4;
            }
        }
    }

  private:
    Teuchos::RCP<const map_type> opMap_;
    const FiberContainer &fc_;
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
    A_fiber_hydro(const global_ordinal_type n, const Teuchos::RCP<const Teuchos::Comm<int>> comm,
                  const FiberContainer &fc, const double eta)
        : fc_(fc), eta_(eta) {
        using Teuchos::rcp;
        TEUCHOS_TEST_FOR_EXCEPTION(comm.is_null(), std::invalid_argument,
                                   "A_fiber_hydro constructor: The input Comm object must be nonnull.");
        if (comm->getRank() == 0) {
            cout << "A_fiber_hydro constructor" << endl;
        }

        const global_ordinal_type indexBase = 0;
        // Construct a map for our block row distribution
        opMap_ = rcp(new map_type(n, indexBase, comm));
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
        const int myRank = comm->getRank();
        if (myRank == 0) {
            cout << "A_fiber_hydro::apply" << endl;
        }

        const size_t numVecs = X.getNumVectors();
        const local_ordinal_type nlocRows = static_cast<local_ordinal_type>(X.getLocalLength());
        // TEUCHOS_TEST_FOR_EXCEPTION(fc_.fibers.size() == nlocRows, std::logic_error, "")
        const size_t nglobCols = opMap_->getGlobalNumElements();
        const size_t baseRow = opMap_->getMinGlobalIndex();
        for (size_t c = 0; c < numVecs; ++c) {
            // Get a view of the desired column
            local_ordinal_type offset = 0;
            Eigen::Map<Eigen::VectorXd> YView(Y.getDataNonConst(c).getRawPtr(), Y.getLocalLength());
            Eigen::VectorXd XEigen = Eigen::Map<const Eigen::VectorXd>(X.getData(0).getRawPtr(), X.getLocalLength());
            Eigen::MatrixXd fw = fc_.apply_fiber_force(XEigen);
            Eigen::MatrixXd v_fib = fc_.flow(fw, eta_);
            YView = fc_.matvec(XEigen, v_fib);
        }
    }

  private:
    Teuchos::RCP<const map_type> opMap_;
    const FiberContainer &fc_;
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
        const int nfibs_per_rank = nfibs_tot / size;
        const int n_pts = 32;
        const int n_time = 10;
        const double eta = 10.0;
        const double bending_rigidity = 0.1;
        const double length = 1.0;
        double dt = 1E-4;
        const int ifib_low = rank * nfibs_per_rank;
        const int ifib_high = (rank + 1) * nfibs_per_rank;

        assert(nfibs_tot % size == 0);
        FiberContainer fibs(nfibs_per_rank, n_pts, bending_rigidity, eta);

        if (rank == 0)
            cout << "Reading in " << nfibs_tot << " fibers.\n";

        for (int ifib = 0; ifib < nfibs_tot; ++ifib) {
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

        A_fiber_hydro::global_ordinal_type n = comm->getSize() * nfibs_per_rank * n_pts * 4;
        RCP<A_fiber_hydro> A_sim = rcp(new A_fiber_hydro(n, comm, fibs, eta));
        RCP<P_inv_hydro> preconditioner = rcp(new P_inv_hydro(n, comm, fibs));
        RCP<const Tpetra::Map<>> map = A_sim->getDomainMap();
        typedef Tpetra::Vector<A_fiber_hydro::scalar_type, A_fiber_hydro::local_ordinal_type,
                               A_fiber_hydro::global_ordinal_type, A_fiber_hydro::node_type>
            vec_type;

        // Create initial vectors
        RCP<vec_type> X, B;
        X = rcp(new vec_type(map));
        B = rcp(new vec_type(map));
        X->putScalar(0.0);

        { // Initialize FMM for reliable benchmarks
            double tmp = fibs.fibers[0].x_(0, 0);
            fibs.fibers[0].x_(0, 0) = 1.0;
            A_sim->apply(*X, *B);
            fibs.fibers[0].x_(0, 0) = tmp;
        }

        { // Initialize RHS
            int offset = 0;
            for (auto &fib : fibs.fibers) {
                std::memcpy(B->getDataNonConst(0).getRawPtr() + offset, fib.RHS_.data(), fib.RHS_.size());
                offset += fib.RHS_.size();
            }
        }

        Belos::LinearProblem<ST, MV, OP> problem(A_sim, X, B);

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

        if (prec_flag)
            problem.setRightPrec(preconditioner);

        bool set = problem.setProblem();
        if (set == false) {
            cout << endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << endl;
            return -1;
        }

        ParameterList belosList;
        belosList.set("Block Size", blocksize);      // Blocksize to be used by iterative solver
        belosList.set("Maximum Iterations", 100);    // Maximum number of iterations allowed
        belosList.set("Convergence Tolerance", tol); // Relative convergence tolerance requested
        belosList.set("Orthogonalization", ortho);   // Orthogonalization type

        Belos::BlockGmresSolMgr<ST, MV, OP> solver(rcpFromRef(problem), rcpFromRef(belosList));
        double st = omp_get_wtime();
        Belos::ReturnType ret = solver.solve();
        if (rank == 0)
            cout << solver.getNumIters() << " " << omp_get_wtime() - st << endl;

        success = true;
        cout << "Test passed\n";
    }
    // TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

    return (success ? EXIT_SUCCESS : EXIT_FAILURE);
}
