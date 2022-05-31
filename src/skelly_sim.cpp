#include <skelly_sim.hpp>

#include <system.hpp>

#include <filesystem>
#include <spdlog/spdlog.h>
#include <stdexcept>

#include <Teuchos_CommandLineProcessor.hpp>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);

    std::string config_file = "skelly_config.toml";
    bool resume_flag = false;
    bool overwrite_flag = false;
    bool post_process_flag = false;
    Teuchos::CommandLineProcessor cmdp(false, true);
    cmdp.setOption("config-file", &config_file, "TOML input file.");
    cmdp.setOption("resume", "no-resume", &resume_flag, "Supply to resume simulation.");
    cmdp.setOption("overwrite", "no-overwrite", &overwrite_flag, "Supply to overwrite existing simulation.");
    cmdp.setOption("post-process", "no-post-process", &post_process_flag, "Supply to post-process existing sim.");
    if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    try {
        if (resume_flag && overwrite_flag)
            throw std::runtime_error("Can't resume and overwrite simultaneously.");
        namespace fs = std::filesystem;
        if (post_process_flag) {
            if (!fs::exists(fs::path{"skelly_sim.out"}))
                throw std::runtime_error("No trajectory detected for post-process.");
            if (resume_flag)
                throw std::runtime_error("--post-process and --resume are mutually exclusive.");
            if (overwrite_flag)
                throw std::runtime_error("--post-process and --overwrite are mutually exclusive.");
        }
        else if (resume_flag) {
            if (!fs::exists(fs::path{"skelly_sim.out"}))
                throw std::runtime_error("--resume flag supplied without existing trajectory.");
        }
        else {
            if (!overwrite_flag && (fs::exists(fs::path{"skelly_sim.out"}) || fs::exists(fs::path{"skelly_sim.vf"})))
                throw std::runtime_error("Existing trajectory detected. Supply --overwrite flag to overwrite.");
        }

        System::init(config_file, resume_flag, post_process_flag);
        if (post_process_flag)
            System::run_post_process();
        else
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
