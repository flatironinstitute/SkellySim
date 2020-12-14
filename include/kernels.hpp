#ifndef KERNELS_HPP
#define KERNELS_HPP
#include <Eigen/Dense>
#include <STKFMM/STKFMM.hpp>

namespace kernels {
typedef Eigen::MatrixXd (*fmm_kernel_func_t)(const int n_trg, const Eigen::Ref<const Eigen::MatrixXd> &f_sl,
                                             const Eigen::Ref<const Eigen::MatrixXd> &f_dl, stkfmm::STKFMM *);

Eigen::MatrixXd oseen_tensor_contract_direct(const Eigen::Ref<const Eigen::MatrixXd> &r_src,
                                             const Eigen::Ref<const Eigen::MatrixXd> &r_trg,
                                             const Eigen::Ref<const Eigen::MatrixXd> &density, double eta = 1.0,
                                             double reg = 5E-3, double epsilon_distance = 1E-5);

Eigen::MatrixXd stokes_vel_fmm(const int n_trg, const Eigen::Ref<const Eigen::MatrixXd> &f_sl,
                               const Eigen::Ref<const Eigen::MatrixXd> &f_dl, stkfmm::STKFMM *fmmPtr);

Eigen::MatrixXd stokes_pvel_fmm(const int n_trg, const Eigen::Ref<const Eigen::MatrixXd> &f_sl,
                                const Eigen::Ref<const Eigen::MatrixXd> &f_dl, stkfmm::STKFMM *fmmPtr);

Eigen::MatrixXd oseen_tensor_direct(const Eigen::Ref<const Eigen::MatrixXd> &r_src,
                                    const Eigen::Ref<const Eigen::MatrixXd> &r_trg, double eta = 1.0, double reg = 5E-3,
                                    double epsilon_distance = 1E-5);

template <typename stkfmm_type>
class FMM {
  public:
    template <typename F>
    FMM(const int order, const int maxPoints, const stkfmm::PAXIS paxis, const stkfmm::KERNEL k, const F &kernel_func)
        : fmmPtr_(new stkfmm_type(order, maxPoints, paxis, static_cast<unsigned>(k))), k_(k),
          kernel_func_(kernel_func){};

    Eigen::MatrixXd operator()(const Eigen::Ref<const Eigen::MatrixXd> &r_sl,
                               const Eigen::Ref<const Eigen::MatrixXd> &r_dl,
                               const Eigen::Ref<const Eigen::MatrixXd> &r_trg,
                               const Eigen::Ref<const Eigen::MatrixXd> &f_sl,
                               const Eigen::Ref<const Eigen::MatrixXd> &f_dl) {
        // Check if LOCAL source/target points have changed, and then broadcast that for a GLOBAL update
        char setup_flag_local =
            (r_sl_old_.size() != r_sl.size() || r_dl_old_.size() != r_dl.size() || r_trg_old_.size() != r_trg.size() ||
             r_sl_old_ != r_sl || r_dl_old_ != r_dl || r_trg_old_ != r_trg);
        char setup_flag;
        MPI_Allreduce(&setup_flag_local, &setup_flag, 1, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);

        if (setup_flag) {
            double sl_min = r_sl.size() ? r_sl.minCoeff() : std::numeric_limits<double>::max();
            double dl_min = r_dl.size() ? r_dl.minCoeff() : std::numeric_limits<double>::max();
            double trg_min = r_trg.size() ? r_trg.minCoeff() : std::numeric_limits<double>::max();
            double sl_max = r_sl.size() ? r_sl.maxCoeff() : std::numeric_limits<double>::min();
            double dl_max = r_dl.size() ? r_dl.maxCoeff() : std::numeric_limits<double>::min();
            double trg_max = r_trg.size() ? r_trg.maxCoeff() : std::numeric_limits<double>::min();

            // Find most extreme points to define our box, which is required to be a cube
            double local_min = std::min(std::min(sl_min, dl_min), trg_min);
            double local_max = std::max(std::max(sl_max, dl_max), trg_max);

            double global_min, global_max;
            MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            // Scale box coordinates so that no source/target lies on the box boundary
            global_min *= 1.01;
            global_max *= 1.01;
            const double L = global_max - global_min;

            // Update FMM tree and cache coordinates
            double origin[3] = {global_min, global_min, global_min};
            fmmPtr_->setBox(origin, L);
            fmmPtr_->setPoints(r_sl.size() / 3, r_sl.data(), r_trg.size() / 3, r_trg.data(), r_dl.size() / 3,
                               r_dl.data());
            fmmPtr_->setupTree(k_);
            r_sl_old_ = r_sl;
            r_dl_old_ = r_dl;
            r_trg_old_ = r_trg;
        }

        int n_trg = r_trg.size() / 3;
        return kernel_func_(n_trg, f_sl, f_dl, fmmPtr_.get());
    }

  private:
    std::unique_ptr<stkfmm::STKFMM> fmmPtr_;
    Eigen::MatrixXd r_sl_old_;
    Eigen::MatrixXd r_dl_old_;
    Eigen::MatrixXd r_trg_old_;
    stkfmm::KERNEL k_;
    fmm_kernel_func_t kernel_func_;
};
}; // namespace kernels

#endif
