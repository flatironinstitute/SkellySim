#include <skelly_sim.hpp>

#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <fiber_container_base.hpp>
#include <fiber_container_finite_difference.hpp>
#include <fiber_finite_difference.hpp>
#include <params.hpp>
#include <system.hpp>

#include <Teuchos_CommandLineProcessor.hpp>
#include <cnpy.hpp>

int main(int argc, char *argv[]) {
    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

        // Only used with finite difference fibers (for now)
        auto &fc = *System::get_fiber_container();
        if (fc.fiber_type_ != FiberContainerBase::FIBERTYPE::FiniteDifference) {
            throw std::runtime_error("dynamic_instability_test only compatible with FiniteDifferenceFiber(s) for now");
        }
        FiberContainerFiniteDifference *fc_fd = static_cast<FiberContainerFiniteDifference *>(&fc);

        std::vector<float> times;
        std::vector<int> n_fibers;
        std::vector<double> lengths;

        double t = 0.0;
        while (t < t_final) {
            times.push_back(t);
            n_fibers.push_back(fc_fd->get_global_fiber_number());
            double length =
                std::accumulate(fc_fd->fibers_.begin(), fc_fd->fibers_.end(), 0.0,
                                [](const double &a, const FiberFiniteDifference &b) { return a + b.length_; });
            double length_tot;
            MPI_Reduce(&length, &length_tot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            lengths.push_back(length_tot / n_fibers.back());
            System::dynamic_instability();
            t += properties.dt;
        }

        if (rank == 0) {
            cnpy::npz_save("di_traj.npz", "t", times.data(), {times.size()}, "w");
            cnpy::npz_save("di_traj.npz", "n_fibers", n_fibers.data(), {n_fibers.size()}, "a");
            cnpy::npz_save("di_traj.npz", "lengths", lengths.data(), {lengths.size()}, "a");
        }
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
