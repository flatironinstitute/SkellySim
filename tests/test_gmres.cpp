#include <skelly_sim.hpp>

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

#include <cnpy.hpp>
#include <fiber.hpp>
#include <periphery.hpp>
#include <solver_hydro.hpp>
#include <system.hpp>
#include <utils.hpp>

#include <spdlog/spdlog.h>
#include <mpi.h>

using namespace Teuchos;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

VectorXd load_vec(cnpy::npz_t &npz, const char *var) { return VectorMap(npz[var].data<double>(), npz[var].shape[0]); }

int main(int argc, char *argv[]) {
    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);
    // Tpetra::ScopeGuard tpetraScope(&argc, &argv);
    bool success = true;
    {
        std::string config_file = "test_gmres.toml";

        CommandLineProcessor cmdp(false, true);
        cmdp.setOption("config-file", &config_file, "TOML input file.");
        if (cmdp.parse(argc, argv) != CommandLineProcessor::PARSE_SUCCESSFUL) {
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        try {
            System::init(config_file);
            System::run();
        }
        catch (...) {
            success = false;
        }

        if (success)
            spdlog::info("Test passed");
    }

    MPI_Finalize();
    return (success ? EXIT_SUCCESS : EXIT_FAILURE);
}
