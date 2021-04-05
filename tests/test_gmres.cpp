#include <skelly_sim.hpp>
#include <system.hpp>

#include <Teuchos_CommandLineProcessor.hpp>

#include <mpi.h>
#include <spdlog/spdlog.h>

int main(int argc, char *argv[]) {
    std::string config_file = "test_gmres.toml";

    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);

    bool success = false;

    Teuchos::CommandLineProcessor cmdp(false, true);
    cmdp.setOption("config-file", &config_file, "TOML input file.");
    if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    try {
        System::init(config_file);
        System::step();
        success = true;
    } catch (std::exception &e) {
        spdlog::info(e.what());
    } catch (...) {
        spdlog::info("Unknown exception detected.");
    }

    if (success)
        spdlog::info("Test passed");

    MPI_Finalize();
    return (success ? EXIT_SUCCESS : EXIT_FAILURE);
}
