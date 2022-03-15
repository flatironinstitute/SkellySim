#include <skelly_sim.hpp>

#include <system.hpp>

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

    System::init(config_file, resume_flag);
    System::prep_state_for_solver();
    System::write();

    auto [fib_sol_size, shell_sol_size, body_sol_size] = System::get_local_solution_sizes();
    int sol_size = fib_sol_size + shell_sol_size + body_sol_size;
    Eigen::VectorXd x(sol_size);
    x.setOnes();
    Eigen::VectorXd b = System::apply_matvec(x);

    Eigen::VectorXd bpy(sol_size);
    bpy.segment(0, shell_sol_size) = b.segment(fib_sol_size, shell_sol_size);
    bpy.segment(shell_sol_size, body_sol_size) = b.segment(fib_sol_size + shell_sol_size, body_sol_size);
    bpy.segment(shell_sol_size + body_sol_size, fib_sol_size) = b.segment(0, fib_sol_size);

    std::cout << bpy.transpose() << std::endl;

    MPI_Finalize();
    return EXIT_SUCCESS;
}
