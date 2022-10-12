#include <kernels.hpp>
#include <omp.h>

int main(int argc, char *argv[]) {
    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);

    using Eigen::MatrixXd;

    constexpr int n_src = 1000;
    constexpr int n_trg = 1000;
    constexpr double eta = 1.0;

    const int mult_order = 8;
    const int max_pts = 50;
    auto stresslet_kernel_fmm = kernels::FMM<stkfmm::Stk3DFMM>(mult_order, max_pts, stkfmm::PAXIS::NONE,
                                                               stkfmm::KERNEL::PVel, kernels::stokes_pvel_fmm);
    auto stokeslet_kernel_fmm = kernels::FMM<stkfmm::Stk3DFMM>(mult_order, max_pts, stkfmm::PAXIS::NONE,
                                                               stkfmm::KERNEL::Stokes, kernels::stokes_vel_fmm);
    {
        MatrixXd r_src = MatrixXd::Random(3, n_src);
        MatrixXd f_src = MatrixXd::Random(3, n_src);
        MatrixXd r_trg = MatrixXd::Random(3, n_trg);
        MatrixXd nullmat;
        omp_set_num_threads(1);
        MatrixXd res_single = kernels::stokeslet_direct_cpu(r_src, nullmat, r_trg, f_src, nullmat, eta);
        omp_set_num_threads(5);
        MatrixXd res_openmp = kernels::stokeslet_direct_cpu(r_src, nullmat, r_trg, f_src, nullmat, eta);
        MatrixXd res_gpu = kernels::stokeslet_direct_gpu(r_src, nullmat, r_trg, f_src, nullmat, eta);
        MatrixXd res_fmm = stokeslet_kernel_fmm(r_src, nullmat, r_trg, f_src, nullmat, eta);

        std::cout << (res_openmp - res_single).mean() << std::endl;
        std::cout << (res_openmp - res_gpu).mean() << std::endl;
        std::cout << (res_single - res_gpu).mean() << std::endl;
        std::cout << (res_single - res_fmm).mean() << std::endl;
    }
    {
        MatrixXd r_src = MatrixXd::Random(3, n_src);
        MatrixXd f_src = MatrixXd::Random(9, n_src);
        MatrixXd r_trg = MatrixXd::Random(3, n_trg);
        MatrixXd nullmat;

        omp_set_num_threads(1);
        MatrixXd res_single = kernels::stresslet_direct_cpu(nullmat, r_src, r_trg, nullmat, f_src, eta);
        omp_set_num_threads(5);
        MatrixXd res_openmp = kernels::stresslet_direct_cpu(nullmat, r_src, r_trg, nullmat, f_src, eta);
        MatrixXd res_gpu = kernels::stresslet_direct_gpu(nullmat, r_src, r_trg, nullmat, f_src, eta);
        MatrixXd res_fmm = stresslet_kernel_fmm(nullmat, r_src, r_trg, nullmat, f_src, eta);

        std::cout << (res_openmp - res_single).mean() << std::endl;
        std::cout << (res_openmp - res_gpu).mean() << std::endl;
        std::cout << (res_single - res_gpu).mean() << std::endl;
        std::cout << (res_single - res_fmm).mean() << std::endl;
    }
    return EXIT_SUCCESS;
}
