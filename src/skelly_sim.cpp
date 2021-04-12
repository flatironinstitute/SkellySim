#include <skelly_sim.hpp>

#include <system.hpp>

#include <mpi.h>
#include <spdlog/spdlog.h>

int main(int argc, char *argv[]) {
    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);

    if (argc != 2) {
        spdlog::critical("No config file provided. Add config_file as argument to skelly_sim.");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    System::init(std::string(argv[1]));
    System::run();

    MPI_Finalize();
    return EXIT_SUCCESS;
}
