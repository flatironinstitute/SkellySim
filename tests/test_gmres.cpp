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
            using Eigen::Map;

            // Current position in result vector
            local_ordinal_type offset = 0;
            // Loop through fibers and apply their preconditioner to the Tpetra result vector
            // Fixme: move Fiber loops to FiberContainer
            for (auto &fib : fc_.fibers) {
                Map<const VectorXd> XView(X.getData(c).getRawPtr() + offset, fib.n_nodes_ * 4);
                Map<VectorXd> res_fib(Y.getDataNonConst(c).getRawPtr() + offset, fib.n_nodes_ * 4);
                res_fib = fib.A_LU_.solve(XView);

                offset += fib.n_nodes_ * 4;
            }

            // Each MPI rank has only a _local_ portion of the inverse matrix, but needs the _global_ 'X' vector to
            // contract against. Each rank will get a full copy of the 'X' in x_shell
            VectorXd x_shell(3 * shell_.n_nodes_global_); /// Shell 'x' across _all_ mpi ranks

            // Collect local 'X' data and distribute to _all_ MPI ranks
            MPI_Allgatherv(X.getData(c).getRawPtr() + offset, shell_.node_counts_[rank], MPI_DOUBLE, x_shell.data(),
                           shell_.node_counts_.data(), shell_.node_displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);

            // Copy local result into local portion of Tpetra result vector via an Eigen map
            Map<VectorXd> res_view_shell(Y.getDataNonConst(c).getRawPtr() + offset, shell_.node_counts_[rank]);
            res_view_shell = shell_.M_inv_ * x_shell;

            offset += x_shell.size();

            // Fixme: move Body loops to BodyContainer
            if (rank == 0) {
                for (int i = 0; i < bc_.get_local_count(); ++i) {
                    const Body &body = bc_.bodies[i];
                    const int n_nodes = body.n_nodes_;
                    Map<const VectorXd> XView(X.getData(c).getRawPtr() + offset, n_nodes * 3 + 6);
                    Map<VectorXd> res_body(Y.getDataNonConst(c).getRawPtr() + offset, n_nodes * 3 + 6);
                    res_body = body.A_LU_.solve(XView);

                    offset += n_nodes * 3 + 6;
                }
            }
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

        const int fib_sol_size = fc_.get_local_solution_size();
        const int shell_sol_size = shell_.get_local_solution_size();
        const int body_sol_size = bc_.get_local_solution_size();
        const int sol_size = fib_sol_size + shell_sol_size + body_sol_size;

        const int fib_sol_offset = 0;
        const int shell_sol_offset = fib_sol_size;
        const int body_sol_offset = shell_sol_offset + shell_sol_size;

        const int fib_v_size = fc_.get_local_node_count();
        const int shell_v_size = shell_.get_local_node_count();
        const int body_v_size = bc_.get_local_node_count();
        const int v_size = fib_v_size + shell_v_size + body_v_size;

        const int fib_v_offset = 0;
        const int shell_v_offset = fib_v_size;
        const int body_v_offset = shell_v_offset + shell_v_size;

        const int n_bodies_global = bc_.get_global_count();

        for (size_t c = 0; c < X.getNumVectors(); ++c) {
            using Eigen::Block;
            using Eigen::Map;

            // Get views and temporary arrays
            double *res_ptr = Y.getDataNonConst(c).getRawPtr();
            const double *x_ptr = X.getData(c).getRawPtr();
            Map<const VectorXd> x_fib_local(x_ptr, fib_sol_size);
            Map<const VectorXd> x_shell_local(x_ptr + shell_sol_offset, shell_sol_size);
            Map<const VectorXd> x_body_local(x_ptr + body_sol_offset, body_sol_size);
            Map<VectorXd> res_fib(res_ptr, fib_sol_size);
            Map<VectorXd> res_shell(res_ptr + fib_sol_size, shell_sol_size);
            Map<VectorXd> res_bodies(res_ptr + fib_sol_size + shell_sol_size, body_sol_size);

            MatrixXd r_all(3, fib_v_size + shell_v_size + body_v_size);
            Block<MatrixXd> r_fib = r_all.block(0, 0, 3, fib_v_size);
            Block<MatrixXd> r_shell = r_all.block(0, shell_v_offset, 3, shell_v_size);
            Block<MatrixXd> r_body = r_all.block(0, body_v_offset, 3, body_v_size);
            r_fib = fc_.get_r_vectors();
            r_shell = shell_.get_node_positions();
            r_body = bc_.get_local_node_positions();
            MatrixXd v_all = MatrixXd(3, v_size);
            Block<MatrixXd> r_shellbody = r_all.block(0, shell_v_offset, 3, shell_v_size + body_v_size);

            Block<MatrixXd> v_fib = v_all.block(0, fib_v_offset, 3, fib_v_size);
            Block<MatrixXd> v_shell = v_all.block(0, shell_v_offset, 3, shell_v_size);
            Block<MatrixXd> v_shellbodies = v_all.block(0, shell_v_offset, 3, shell_v_size + body_v_size);
            Block<MatrixXd> v_bodies = v_all.block(0, body_v_offset, 3, body_v_size);

            // Collect all X shell data into single vector on all ranks
            // (for res_shell_local = A_shell_local * x_shell_global + ...)
            VectorXd x_shell_global(3 * shell_.n_nodes_global_);
            MPI_Allgatherv(x_shell_local.data(), shell_sol_size, MPI_DOUBLE, x_shell_global.data(),
                           shell_.node_counts_.data(), shell_.node_displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);

            Eigen::MatrixXd body_velocities, body_densities;
            std::tie(body_velocities, body_densities) = bc_.unpack_solution_vector(x_body_local);

            // calculate fiber-fiber velocity
            MatrixXd fw = fc_.apply_fiber_force(x_fib_local);
            MatrixXd v_fib2all = fc_.flow(fw, r_shellbody, eta_);
            Block<MatrixXd> v_fib2fib = v_fib2all.block(0, fib_v_offset, 3, fib_v_size);
            Block<MatrixXd> v_fib2shell = v_fib2all.block(0, shell_v_offset, 3, shell_v_size);
            Block<MatrixXd> v_fib2bodies = v_fib2all.block(0, body_v_offset, 3, body_v_size);

            MatrixXd r_fibbody(3, r_fib.cols() + r_body.cols());
            r_fibbody.block(0, 0, 3, r_fib.cols()) = r_fib;
            r_fibbody.block(0, r_fib.cols(), 3, r_body.cols()) = r_body;
            MatrixXd v_shell2fibbody = shell_.flow(r_fibbody, x_shell_local, eta_);

            v_all = v_fib2all;
            v_fib += v_shell2fibbody.block(0, 0, 3, r_fib.cols());
            v_bodies += v_shell2fibbody.block(0, r_fib.cols(), 3, r_body.cols());

            // Calculate forces/torques on body
            if (bc_.bodies.size()) {
                MatrixXd force_torque_bodies, v_fib_boundary;
                std::tie(force_torque_bodies, v_fib_boundary) =
                    System::calculate_body_fiber_link_conditions(fc_, bc_, x_fib_local, body_velocities);
                MPI_Allreduce(MPI_IN_PLACE, force_torque_bodies.data(), force_torque_bodies.size(), MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);

                // Calculate flow on other objects due to body forces
                MatrixXd v_bdy2all = bc_.flow(r_all, body_densities, force_torque_bodies, eta_);
                Block<MatrixXd> v_bdy2fib = v_bdy2all.block(0, fib_v_offset, 3, fib_v_size);
                Block<MatrixXd> v_bdy2shell = v_bdy2all.block(0, shell_v_offset, 3, shell_v_size);
                Block<MatrixXd> v_bdy2bdy = v_bdy2all.block(0, body_v_offset, 3, body_v_size);
                v_all += v_bdy2all;

                if (rank == 0) {
                    int node_offset = 0;
                    int i_body = 0;
                    for (auto &body : bc_.bodies) {
                        Map<VectorXd> res_body_nodes(res_bodies.data(), body.n_nodes_ * 3);
                        Map<VectorXd> res_body_com(res_bodies.data() + body.n_nodes_ * 3, 6);

                        Block<MatrixXd> d = body_densities.block(0, node_offset, 3, body.n_nodes_);
                        Block<MatrixXd> U = body_velocities.block(0, i_body, 6, 1);

                        VectorXd cx(3 * body.n_nodes_), cy(3 * body.n_nodes_), cz(3 * body.n_nodes_);
                        for (int i = 0; i < body.n_nodes_; ++i) {
                            cx.segment(i * 3, 3) += d(0, i) / body.node_weights_(i) * body.ex_.col(i);
                            cy.segment(i * 3, 3) += d(1, i) / body.node_weights_(i) * body.ey_.col(i);
                            cz.segment(i * 3, 3) += d(2, i) / body.node_weights_(i) * body.ez_.col(i);
                        }

                        VectorXd KU = body.K_ * U;
                        VectorXd KTLambda = body.K_.transpose() * Map<VectorXd>(d.data(), 3 * body.n_nodes_);

                        res_body_nodes +=
                            -(cx + cy + cz) - KU + Map<VectorXd>(v_bodies.data() + node_offset * 3, body.n_nodes_ * 3);
                        res_body_com = -KTLambda + U;

                        i_body++;
                        node_offset += body.n_nodes_;
                    }
                }
            }

            res_fib = fc_.matvec(x_fib_local, v_fib);
            res_shell =
                shell_.stresslet_plus_complementary_ * x_shell_global + Map<VectorXd>(v_shell.data(), shell_sol_size);
        }
    }

  private:
    RCP<const map_type> opMap_;
    const FiberContainer &fc_;
    const Periphery &shell_;
    const BodyContainer &bc_;
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
            MatrixXd r_trg_external = shell.node_pos_;
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

            shell.update_RHS(v_fib2all.block(0, offset, 3, r_trg_external.cols()));

            // FIXME: Body update_RHS
            bc.update_cache_variables(eta);
            Eigen::MatrixXd v_on_bodies = Eigen::MatrixXd::Zero(3, bc.get_local_solution_size());
            bc.update_RHS(v_on_bodies);
        }

        RCP<A_fiber_hydro> A_sim = rcp(new A_fiber_hydro(comm, eta));
        RCP<P_inv_hydro> preconditioner = rcp(new P_inv_hydro(comm));

        RCP<const Tpetra::Map<>> map = A_sim->getDomainMap();
        typedef Tpetra::Vector<A_fiber_hydro::scalar_type, A_fiber_hydro::local_ordinal_type,
                               A_fiber_hydro::global_ordinal_type, A_fiber_hydro::node_type>
            vec_type;

        // Create initial vectors
        RCP<vec_type> X, RHS;
        X = rcp(new vec_type(map));
        RHS = rcp(new vec_type(map));

        X->putScalar(0.0);
        RHS->putScalar(0.0);
        const int fib_sol_size = fc.get_local_solution_size();
        const int shell_sol_size = shell.get_local_solution_size();
        const int body_sol_size = bc.get_local_solution_size();
        Eigen::Map<Eigen::VectorXd> RHS_fib(RHS->getDataNonConst(0).getRawPtr(), fib_sol_size);
        Eigen::Map<Eigen::VectorXd> RHS_shell(RHS->getDataNonConst(0).getRawPtr() + fib_sol_size, shell_sol_size);
        Eigen::Map<Eigen::VectorXd> RHS_body(RHS->getDataNonConst(0).getRawPtr() + fib_sol_size + shell_sol_size,
                                             body_sol_size);

        // Initialize GMRES RHS vector
        RHS_fib = fc.get_RHS();
        RHS_shell = shell.get_RHS();
        RHS_body = bc.get_RHS();

        // Output application of A_hydro operator on simple input for comparison to python output
        {
            RCP<vec_type> Y = rcp(new vec_type(map));
            Eigen::Map<Eigen::VectorXd> fib_Y(Y->getDataNonConst(0).getRawPtr(), fib_sol_size);
            Eigen::Map<Eigen::VectorXd> shell_Y(Y->getDataNonConst(0).getRawPtr() + fib_sol_size, shell_sol_size);

            X->putScalar(1.0);
            A_sim->apply(*X, *Y);
            X->putScalar(0.0);

            Eigen::VectorXd RHS_fib_global = utils::collect_into_global(RHS_fib);
            Eigen::VectorXd RHS_shell_global = utils::collect_into_global(RHS_shell);
            Eigen::VectorXd fib_Y_global = utils::collect_into_global(fib_Y);
            Eigen::VectorXd shell_Y_global = utils::collect_into_global(shell_Y);

            if (rank == 0) {
                cnpy::npy_save("RHS_fib.npy", RHS_fib_global.data(), {(unsigned long)RHS_fib_global.size()});
                cnpy::npy_save("RHS_shell.npy", RHS_shell_global.data(), {(unsigned long)RHS_shell_global.size()});
                cnpy::npy_save("Y_fib.npy", fib_Y_global.data(), {(unsigned long)fib_Y_global.size()});
                cnpy::npy_save("Y_shell.npy", shell_Y_global.data(), {(unsigned long)RHS_shell_global.size()});
            }
        }

        fc.fmm_->force_setup_tree();
        shell.fmm_->force_setup_tree();

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
