#include <skelly_sim.hpp>

#include <system.hpp>

#include <spdlog/spdlog.h>
#include <stdexcept>

#include <Teuchos_CommandLineProcessor.hpp>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);

    std::string config_file;
    int resume_flag = false;
    Teuchos::CommandLineProcessor cmdp(false, true);
    cmdp.setOption("config-file", &config_file, "TOML input file.");
    cmdp.setOption("resume-flag", &resume_flag, "Flag to resume simulation.");
    if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    try {
        System::init(config_file, resume_flag);
        System::run();
    } catch (const std::runtime_error &e) {
        // Warning: Critical only catches things on rank 0, so this may or may not print, if
        // some random rank throws an error. This is the same reason we use MPI_Abort: all
        // ranks are not guaranteed to land here, so there's only so much grace that can be
        // easily implemented.
        spdlog::critical(std::string("Fatal exception caught: ") + e.what());
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
