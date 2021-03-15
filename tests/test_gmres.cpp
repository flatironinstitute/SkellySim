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

#include <spdlog/spdlog.h>

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
    RCP<FancyOStream> fos = fancyOStream(rcpFromRef(cout));
    std::string config_file = "test_gmres.toml";

    CommandLineProcessor cmdp(false, true);
    cmdp.setOption("config-file", &config_file, "TOML input file.");
    if (cmdp.parse(argc, argv) != CommandLineProcessor::PARSE_SUCCESSFUL) {
        return -1;
    }

    System::init(config_file);
    System::step();

    success = true;
    spdlog::info("Test passed");

    return (success ? EXIT_SUCCESS : EXIT_FAILURE);
}
