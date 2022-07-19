#include <skelly_sim.hpp>

#include <fiber.hpp>
#include <params.hpp>
#include <string>
#include <system.hpp>

#include <filesystem>
#include <spdlog/spdlog.h>
#include <stdexcept>

#include <Teuchos_CommandLineProcessor.hpp>
#include <mpi.h>

#include <cnpy.hpp>
#include <unordered_map>

int main(int argc, char *argv[]) {
    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);

    std::string config_file = "skelly_config.toml";
    Teuchos::CommandLineProcessor cmdp(false, true);
    cmdp.setOption("config-file", &config_file, "TOML input file.");
    if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    try {
        System::init(config_file, false, false);
        auto params = System::get_params();
        auto &properties = System::get_properties();
        double t_final = params->t_final;
        auto &fc = *System::get_fiber_container();


        std::vector<float> times;
        std::vector<int> n_fibers;

        double t = 0.0;
        while (t < t_final) {
            times.push_back(t);
            n_fibers.push_back(fc.get_global_count());
            System::dynamic_instability();
            t += properties.dt;
        }

        cnpy::npz_save("di_traj.npz", "t", times.data(), {times.size()}, "w");
        cnpy::npz_save("di_traj.npz", "n_fibers", n_fibers.data(), {n_fibers.size()}, "a");

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
