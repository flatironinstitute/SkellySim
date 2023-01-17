#include <fstream>
#include <skelly_sim.hpp>

#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <params.hpp>
#include <periphery.hpp>
#include <point_source.hpp>
#include <system.hpp>

#include <Teuchos_CommandLineProcessor.hpp>

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
        auto psc = System::get_point_source_container();

        PointSourceContainer psc_old = *psc;

        System::solve();

        psc->points.clear();

        double radius = System::get_shell()->node_pos_.col(0).norm();
        int n_nodes = System::get_shell()->n_nodes_global_;

        Eigen::MatrixXd r_trg = Eigen::MatrixXd::Zero(3, 100);
        r_trg.row(2).setLinSpaced(-radius, radius);

        Eigen::MatrixXd velocity_system = System::velocity_at_targets(r_trg);

        std::ofstream outstream("N" + std::to_string(n_nodes) + "_Rp" + std::to_string(radius) + "_Rs" +
                                std::to_string(psc_old.points[0].position_[2]) + ".dat");
        Eigen::MatrixXd velocity_points = psc_old.flow(r_trg, params->eta, 0.0);
        for (int i = 0; i < r_trg.cols(); ++i)
            outstream << r_trg(2, i) << " "
                      << (1.0 + velocity_system.col(i).array() / velocity_points.col(i).array()).abs()[2] << " "
                      << velocity_points.col(i).transpose()[2] << std::endl;

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
