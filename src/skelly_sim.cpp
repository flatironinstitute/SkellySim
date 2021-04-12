#include <skelly_sim.hpp>

#include <system.hpp>

#include <Teuchos_CommandLineProcessor.hpp>

#include <mpi.h>
#include <spdlog/spdlog.h>

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

    System::init(config_file, resume_flag);
    System::run();

    MPI_Finalize();
    return EXIT_SUCCESS;
}
