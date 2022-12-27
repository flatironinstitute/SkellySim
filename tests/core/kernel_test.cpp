#include <iostream>
#include <omp.h>
#include <unordered_map>

#include <Teuchos_CommandLineProcessor.hpp>
#include <kernels.hpp>

using Eigen::MatrixXd;

int main(int argc, char *argv[]) {
    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);

    std::string kernel, driver;

    Teuchos::CommandLineProcessor cmdp(false, true);
    cmdp.setOption("kernel", &kernel, "kernel to evaluate [stokeslet, stresslet]", true);
    cmdp.setOption("driver", &driver, "driver to evaluate [single, openmp, gpu, fmm]", true);

    if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    constexpr int n_src = 1229;
    constexpr int n_trg = 743;
    constexpr double eta = 1.3;

    const int mult_order = 16;
    const int max_pts = 50;

    MatrixXd r_src = MatrixXd::Random(3, n_src);
    MatrixXd r_trg = MatrixXd::Random(3, n_trg);
    MatrixXd nullmat;
    double err;
    if (kernel == "stokeslet") {
        MatrixXd f_src = MatrixXd::Random(3, n_src);

        omp_set_num_threads(1);
        MatrixXd ref = kernels::stokeslet_direct_cpu(r_src, nullmat, r_trg, f_src, nullmat, eta);
        MatrixXd other;

        if (driver == "single") {
            omp_set_num_threads(1);
            other = kernels::stokeslet_direct_cpu(r_src, nullmat, r_trg, f_src, nullmat, eta);
        } else if (driver == "openmp") {
            omp_set_num_threads(4);
            other = kernels::stokeslet_direct_cpu(r_src, nullmat, r_trg, f_src, nullmat, eta);
        } else if (driver == "gpu") {
            other = kernels::stokeslet_direct_gpu(r_src, nullmat, r_trg, f_src, nullmat, eta);
        } else if (driver == "fmm") {
            auto stokeslet_kernel_fmm = kernels::FMM<stkfmm::Stk3DFMM>(mult_order, max_pts, stkfmm::PAXIS::NONE,
                                                                       stkfmm::KERNEL::Stokes, kernels::stokes_vel_fmm);
            other = stokeslet_kernel_fmm(r_src, nullmat, r_trg, f_src, nullmat, eta);
        } else {
            std::cerr << "Invalid driver supplied \"" + driver << "\"" << std::endl;
            return EXIT_FAILURE;
        }

        err = sqrt((ref - other).squaredNorm());
    } else if (kernel == "stresslet") {
        MatrixXd f_src = MatrixXd::Random(9, n_src);

        omp_set_num_threads(1);
        MatrixXd ref = kernels::stresslet_direct_cpu(nullmat, r_src, r_trg, nullmat, f_src, eta);
        MatrixXd other;

        if (driver == "single") {
            omp_set_num_threads(1);
            other = kernels::stresslet_direct_cpu(nullmat, r_src, r_trg, nullmat, f_src, eta);
        } else if (driver == "openmp") {
            omp_set_num_threads(4);
            other = kernels::stresslet_direct_cpu(nullmat, r_src, r_trg, nullmat, f_src, eta);
        } else if (driver == "gpu") {
            other = kernels::stresslet_direct_gpu(nullmat, r_src, r_trg, nullmat, f_src, eta);
        } else if (driver == "fmm") {
            auto stresslet_kernel_fmm = kernels::FMM<stkfmm::Stk3DFMM>(mult_order, max_pts, stkfmm::PAXIS::NONE,
                                                                       stkfmm::KERNEL::PVel, kernels::stokes_pvel_fmm);
            other = stresslet_kernel_fmm(nullmat, r_src, r_trg, nullmat, f_src, eta);
        } else {
            std::cerr << "Invalid driver supplied \"" + driver << "\"" << std::endl;
            return EXIT_FAILURE;
        }

        err = sqrt((ref - other).squaredNorm());
    } else {
        std::cerr << "Invalid kernel supplied \"" + kernel << "\"" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << err << std::endl;
    return err > 5E-9;
}
