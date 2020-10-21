// ************************************************************************
//
//                 Belos: Block Linear Solvers Package
//                  Copyright 2004 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************

// NOTE: No preconditioner is used in this case.
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>

// I/O for Harwell-Boeing files
#include <Tpetra_MatrixIO.hpp>

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardCatchMacros.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>

#include <fiber.hpp>

using namespace Teuchos;
using std::cout;
using std::endl;
using std::vector;
using Teuchos::tuple;
using Tpetra::CrsMatrix;
using Tpetra::MultiVector;
using Tpetra::Operator;

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
            std::cout << "P_inv_hydro::apply" << std::endl;
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
                  const FiberContainer &fc)
        : fc_(fc) {
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
            std::cout << "A_fiber_hydro::apply" << std::endl;
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
            Eigen::MatrixXd v_fib = fc_.flow(fw);
            YView = fc_.matvec(XEigen, v_fib);
        }
    }

  private:
    Teuchos::RCP<const map_type> opMap_;
    const FiberContainer &fc_;
};

int main(int argc, char *argv[]) {

    typedef double ST;
    typedef ScalarTraits<ST> SCT;
    typedef SCT::magnitudeType MT;
    typedef Tpetra::Operator<ST> OP;
    typedef Tpetra::MultiVector<ST> MV;
    typedef Belos::OperatorTraits<ST, MV, OP> OPT;
    typedef Belos::MultiVecTraits<ST, MV> MVT;

    Tpetra::ScopeGuard tpetraScope(&argc, &argv);

    bool success = false;
    bool verbose = false;
    try {
        const ST one = SCT::one();

        int MyPID = 0;

        RCP<const Comm<int>> comm = Tpetra::getDefaultComm();
        //
        // Get test parameters from command-line processor
        //
        bool proc_verbose = false;
        bool debug = false;
        int frequency = -1;        // how often residuals are printed by solver
        int numrhs = 1;            // total number of right-hand sides to solve for
        int blocksize = 1;         // blocksize used by solver
        int maxiters = -1;         // maximum number of iterations for solver to use
        std::string ortho("DGKS"); // orthogonalization type
        std::string filename("bcsstk14.hb");
        MT tol = 1.0e-5; // relative residual tolerance

        CommandLineProcessor cmdp(false, true);
        cmdp.setOption("verbose", "quiet", &verbose, "Print messages and results.");
        cmdp.setOption("debug", "nodebug", &debug, "Run debugging checks.");
        cmdp.setOption("frequency", &frequency, "Solvers frequency for printing residuals (#iters).");
        cmdp.setOption("tol", &tol, "Relative residual tolerance used by Gmres solver.");
        cmdp.setOption("filename", &filename, "Filename for Harwell-Boeing test matrix.");
        cmdp.setOption("num-rhs", &numrhs, "Number of right-hand sides to be solved for.");
        cmdp.setOption("max-iters", &maxiters,
                       "Maximum number of iterations per linear system (-1 := adapted to problem/block size).");
        cmdp.setOption("block-size", &blocksize, "Block size to be used by the Gmres solver.");
        cmdp.setOption("ortho-type", &ortho, "Orthogonalization type, either DGKS, ICGS or IMGS (or TSQR if enabled)");
        if (cmdp.parse(argc, argv) != CommandLineProcessor::PARSE_SUCCESSFUL) {
            return -1;
        }
        if (debug) {
            verbose = true;
        }
        if (!verbose) {
            frequency = -1; // reset frequency if test is not verbose
        }

        MyPID = rank(*comm);
        proc_verbose = (verbose && (MyPID == 0));

        if (proc_verbose) {
            std::cout << Belos::Belos_Version() << std::endl << std::endl;
        }

        //
        // Get the data from the HB file and build the Map,Matrix
        //
        RCP<CrsMatrix<ST>> A;
        Tpetra::Utils::readHBMatrix(filename, comm, A);
        RCP<const Tpetra::Map<>> map = A->getDomainMap();

        // Create initial vectors
        RCP<MV> B, X;
        X = rcp(new MV(map, numrhs));
        MVT::MvRandom(*X);
        B = rcp(new MV(map, numrhs));
        OPT::Apply(*A, *X, *B);
        MVT::MvInit(*X, 0.0);

        //
        // ********Other information used by block solver***********
        // *****************(can be user specified)******************
        //
        const int NumGlobalElements = B->getGlobalLength();
        if (maxiters == -1) {
            maxiters = NumGlobalElements / blocksize - 1; // maximum number of iterations to run
        }
        //
        ParameterList belosList;
        belosList.set("Block Size", blocksize);        // Blocksize to be used by iterative solver
        belosList.set("Maximum Iterations", maxiters); // Maximum number of iterations allowed
        belosList.set("Convergence Tolerance", tol);   // Relative convergence tolerance requested
        belosList.set("Orthogonalization", ortho);     // Orthogonalization type

        int verbLevel = Belos::Errors + Belos::Warnings;
        if (debug) {
            verbLevel += Belos::Debug;
        }
        if (verbose) {
            verbLevel += Belos::TimingDetails + Belos::FinalSummary + Belos::StatusTestDetails;
        }
        belosList.set("Verbosity", verbLevel);
        if (verbose) {
            if (frequency > 0) {
                belosList.set("Output Frequency", frequency);
            }
        }
        //
        // Construct an unpreconditioned linear problem instance.
        //
        Belos::LinearProblem<ST, MV, OP> problem(A, X, B);
        problem.setInitResVec(B);
        bool set = problem.setProblem();
        if (set == false) {
            if (proc_verbose)
                std::cout << std::endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << std::endl;
            return -1;
        }
        //
        // *******************************************************************
        // *************Start the block Gmres iteration***********************
        // *******************************************************************
        //
        Belos::BlockGmresSolMgr<ST, MV, OP> solver(rcpFromRef(problem), rcpFromRef(belosList));

        //
        // **********Print out information about problem*******************
        //
        if (proc_verbose) {
            std::cout << std::endl << std::endl;
            std::cout << "Dimension of matrix: " << NumGlobalElements << std::endl;
            std::cout << "Number of right-hand sides: " << numrhs << std::endl;
            std::cout << "Block size used by solver: " << blocksize << std::endl;
            std::cout << "Max number of Gmres iterations: " << maxiters << std::endl;
            std::cout << "Relative residual tolerance: " << tol << std::endl;
            std::cout << std::endl;
        }
        //
        // Perform solve
        //
        Belos::ReturnType ret = solver.solve();
        //
        // Compute actual residuals.
        //
        bool badRes = false;
        std::vector<MT> actual_resids(numrhs);
        std::vector<MT> rhs_norm(numrhs);
        MV resid(map, numrhs);
        OPT::Apply(*A, *X, resid);
        MVT::MvAddMv(-one, resid, one, *B, resid);
        MVT::MvNorm(resid, actual_resids);
        MVT::MvNorm(*B, rhs_norm);
        if (proc_verbose) {
            std::cout << "---------- Actual Residuals (normalized) ----------" << std::endl << std::endl;
        }
        for (int i = 0; i < numrhs; i++) {
            MT actRes = actual_resids[i] / rhs_norm[i];
            if (proc_verbose) {
                std::cout << "Problem " << i << " : \t" << actRes << std::endl;
            }
            if (actRes > tol)
                badRes = true;
        }

        success = (ret == Belos::Converged && !badRes);

        if (success) {
            if (proc_verbose)
                std::cout << "\nEnd Result: TEST PASSED" << std::endl;
        } else {
            if (proc_verbose)
                std::cout << "\nEnd Result: TEST FAILED" << std::endl;
        }

        const int n_pts = 8;
        const int n_fib_per_rank = 2;
        const int n_time = 10;

        FiberContainer fibs(n_fib_per_rank, n_pts, 0.1, 1.0);

        for (int i = 0; i < n_fib_per_rank; ++i) {
            fibs.fibers[i].translate({0., 0.,
                                      100 * static_cast<double>(i + n_fib_per_rank * comm->getRank()) /
                                          (comm->getSize() * n_fib_per_rank + 1)});
            fibs.fibers[i].length_ = 1.0;
            fibs.fibers[i].update_derivatives();
            fibs.fibers[i].form_linear_operator(0.005);
            fibs.fibers[i].build_preconditioner();
            fibs.fibers[i].form_force_operator();
            fibs.fibers[i].update_stokeslet(0.005);
        }

        A_fiber_hydro::global_ordinal_type n = comm->getSize() * n_fib_per_rank * n_pts * 4;
        const int rank = comm->getRank();
        A_fiber_hydro K(n, comm, fibs);

        // Construct a Vector of all ones, using the above Operator's
        // domain Map.
        typedef Tpetra::Vector<A_fiber_hydro::scalar_type, A_fiber_hydro::local_ordinal_type,
                               A_fiber_hydro::global_ordinal_type, A_fiber_hydro::node_type>
            vec_type;
        vec_type x(K.getDomainMap());
        x.putScalar(1.0);
        // Construct an output Vector for K(x).
        vec_type y(K.getRangeMap());
        K.apply(x, y); // Compute y := K(x).

        std::cout << y << std::endl;
        Teuchos::ArrayRCP<double> YView = y.getDataNonConst(0);
        for (int i = 0; i < y.getLocalLength(); ++i)
            std::cout << YView[i] << " ";
        std::cout << std::endl;
    }
    TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

    return (success ? EXIT_SUCCESS : EXIT_FAILURE);
}
