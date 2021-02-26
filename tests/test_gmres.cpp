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
#include <solver_hydro.hpp>
#include <system.hpp>
#include <utils.hpp>

using namespace Teuchos;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

VectorXd load_vec(cnpy::npz_t &npz, const char *var) { return VectorMap(npz[var].data<double>(), npz[var].shape[0]); }

int main(int argc, char *argv[]) {
    Tpetra::ScopeGuard tpetraScope(&argc, &argv);

    bool success = false;
    RCP<const Comm<int>> comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank();
    RCP<FancyOStream> fos = fancyOStream(rcpFromRef(cout));

    std::string config_file = "test_gmres.toml";

    CommandLineProcessor cmdp(false, true);
    cmdp.setOption("config-file", &config_file, "TOML input file.");
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

    for (auto &fib : fc.fibers) {
        fib.update_derivatives();
        fib.update_stokeslet(eta);
        fib.update_linear_operator(dt, eta);
    }

    MatrixXd r_trg_external(3, shell.get_local_node_count() + bc.get_local_node_count());
    r_trg_external.block(0, 0, 3, shell.get_local_node_count()) = shell.get_local_node_positions();
    r_trg_external.block(0, shell.get_local_node_count(), 3, bc.get_local_node_count()) = bc.get_local_node_positions();

    MatrixXd f_on_fibers = fc.generate_constant_force();
    MatrixXd v_fib2all = fc.flow(f_on_fibers, r_trg_external, eta);
    size_t offset = 0;
    for (auto &fib : fc.fibers) {
        fib.update_RHS(dt, v_fib2all.block(0, offset, 3, fib.n_nodes_), f_on_fibers.block(0, offset, 3, fib.n_nodes_));
        fib.apply_bc_rectangular(dt, v_fib2all.block(0, offset, 3, fib.n_nodes_),
                                 f_on_fibers.block(0, offset, 3, fib.n_nodes_));
        fib.update_preconditioner();
        fib.update_force_operator();
        offset += fib.n_nodes_;
    }

    shell.update_RHS(v_fib2all.block(0, offset, 3, shell.get_local_node_count()));
    offset += shell.get_local_node_count();

    bc.update_cache_variables(eta);
    bc.update_RHS(v_fib2all.block(0, offset, 3, bc.get_local_node_count()));

    Solver<P_inv_hydro, A_fiber_hydro> solver_;
    solver_.set_RHS();
    solver_.solve();

    success = true;
    if (rank == 0)
        cout << "Test passed\n";

    return (success ? EXIT_SUCCESS : EXIT_FAILURE);
}
