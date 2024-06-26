#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <functional>
#include <skelly_sim.hpp>

#include <Eigen/Dense>
#include <STKFMM/STKFMM.hpp>

/// Namespace for miscellaneous "kernel" functions and related convenience FMM class
namespace kernels {
typedef Eigen::MatrixXd (*fmm_kernel_func_t)(const int n_trg, CMatrixRef &f_sl, CMatrixRef &f_dl, stkfmm::STKFMM *);

using Evaluator = std::function<Eigen::MatrixXd(CMatrixRef &r_sl, CMatrixRef &r_dl, CMatrixRef &r_trg, CMatrixRef &f_sl,
                                                CMatrixRef &f_dl, double eta)>;

void stokeslet_direct_gpu_impl(const double *r_src, const double *f_src, int n_src, const double *r_trg, double *u_trg,
                               int n_trg);
void stresslet_direct_gpu_impl(const double *r_src, const double *f_src, int n_src, const double *r_trg, double *u_trg,
                               int n_trg);

Eigen::MatrixXd stokeslet_direct_cpu(CMatrixRef &r_sl, CMatrixRef &r_dl, CMatrixRef &r_trg, CMatrixRef &f_sl,
                                     CMatrixRef &f_dl, double eta);

Eigen::MatrixXd stresslet_direct_cpu(CMatrixRef &r_sl, CMatrixRef &r_dl, CMatrixRef &r_trg, CMatrixRef &f_sl,
                                     CMatrixRef &f_dl, double eta);

Eigen::MatrixXd stokeslet_direct_gpu(CMatrixRef &r_sl, CMatrixRef &r_dl, CMatrixRef &r_trg, CMatrixRef &f_sl,
                                     CMatrixRef &f_dl, double eta);

Eigen::MatrixXd stresslet_direct_gpu(CMatrixRef &r_sl, CMatrixRef &r_dl, CMatrixRef &r_trg, CMatrixRef &f_sl,
                                     CMatrixRef &f_dl, double eta);

Eigen::MatrixXd oseen_tensor_contract_direct(CMatrixRef &r_src, CMatrixRef &r_trg, CMatrixRef &density, double eta,
                                             double reg = 5E-3, double epsilon_distance = 1E-5);

Eigen::MatrixXd stokes_vel_fmm(const int n_trg, CMatrixRef &f_sl, CMatrixRef &f_dl, stkfmm::STKFMM *fmmPtr);

Eigen::MatrixXd stokes_pvel_fmm(const int n_trg, CMatrixRef &f_sl, CMatrixRef &f_dl, stkfmm::STKFMM *fmmPtr);

Eigen::MatrixXd oseen_tensor_direct(CMatrixRef &r_src, CMatrixRef &r_trg, double eta, double reg = 5E-3,
                                    double epsilon_distance = 1E-5);

Eigen::MatrixXd rotlet(CMatrixRef &r_src, CMatrixRef &r_trg, CMatrixRef &density, double eta, double reg = 5E-3,
                       double epsilon_distance = 1E-5);

Eigen::MatrixXd stresslet_times_normal(CMatrixRef &r_src, CMatrixRef &normals, double eta, double reg = 5E-3,
                                       double epsilon_distance = 1E-5);

Eigen::MatrixXd stresslet_times_normal_times_density(CMatrixRef &r_src, CMatrixRef &normals, CMatrixRef &density,
                                                     double eta, double reg = 5E-3, double epsilon_distance = 1E-5);

/// Convenience class to represent an FMM interaction, which stores the STKFMM pointer. This
/// setup allows for a direct call to the FMM object which returns the relevant target kernel
/// evaluation matrix to each MPI rank.
template <typename stkfmm_type>
class FMM {
  public:
    template <typename F>
    FMM(const int order, const int maxPoints, const stkfmm::PAXIS paxis, const stkfmm::KERNEL k, const F &kernel_func)
        : fmmPtr_(new stkfmm_type(order, maxPoints, paxis, static_cast<unsigned>(k))), k_(k),
          kernel_func_(kernel_func){};

    /// @brief Set flag to force next call to set up tree, regardless of cache variables
    void force_setup_tree() { force_setup_tree_ = true; };

    /// @brief Evaluate the FMM kernel given the given sources/targets
    ///
    /// Repeated calls to the FMM object with the same source/target positions will maintain
    /// the FMM tree and therefore avoid the costly STKFMM::setupTree() call.
    ///
    /// @param[in] r_sl [ 3 x n_src ] matrix of 'single-layer' source coordinates
    /// @param[in] r_dl [ 3 x n_src ] matrix of 'double-layer' source coordinates
    /// @param[in] r_trg [ 3 x n_trg ] matrix of target coordinates
    /// @param[in] f_sl [ k_dim_sl x n_src ] matrix of 'single-layer' source strengths
    /// @param[in] f_sl [ k_dim_dl x n_src ] matrix of 'double-layer' source strengths
    /// @returns [ k_dim_trg x n_trg ] matrix of kernel evaluated at target positions given the sources
    Eigen::MatrixXd operator()(CMatrixRef &r_sl, CMatrixRef &r_dl, CMatrixRef &r_trg, CMatrixRef &f_sl,
                               CMatrixRef &f_dl, double eta) {
        // Check if LOCAL source/target points have changed, and then broadcast that for a GLOBAL update
        char setup_flag_local =
            (force_setup_tree_ || r_sl_old_.size() != r_sl.size() || r_dl_old_.size() != r_dl.size() ||
             r_trg_old_.size() != r_trg.size() || r_sl_old_ != r_sl || r_dl_old_ != r_dl || r_trg_old_ != r_trg);
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
            force_setup_tree_ = false;
        }

        int n_trg = r_trg.size() / 3;
        return kernel_func_(n_trg, f_sl, f_dl, fmmPtr_.get()) / eta;
    }

  private:
    std::shared_ptr<stkfmm::STKFMM> fmmPtr_; ///< Pointer to underlying STKFMM object
    bool force_setup_tree_ = true; ///< When set, forces tree to rebuild on next call, then is cleared. Useful for
                                   ///< testing/benchmarking
    Eigen::MatrixXd r_sl_old_;     ///< cached 'single-layer' source positions to check for FMM tree invalidation
    Eigen::MatrixXd r_dl_old_;     ///< cache 'double-layer' source positions to check for FMM tree invalidation
    Eigen::MatrixXd r_trg_old_;    ///< cache target positions to check for FMM tree invalidation
    stkfmm::KERNEL k_;             ///< Kernel enum from STKFMM that this interaction calls
    fmm_kernel_func_t
        kernel_func_; ///< Kernel function pointer from our own kernels namespace for the kernel this object will call
};

/// Convenience class to represent GPU all-pairs interaction.
class GPUEvaluator {
  public:
    GPUEvaluator(const Evaluator &kernel_func) : kernel_func_(kernel_func){};

    /// @brief Set flag to force next call to set up tree, regardless of cache variables
    void force_device_sync() { force_device_sync_ = true; };

    /// @brief Evaluate the kernel given the given sources/targets
    ///
    /// @param[in] r_sl [ 3 x n_src ] matrix of 'single-layer' source coordinates
    /// @param[in] r_dl [ 3 x n_src ] matrix of 'double-layer' source coordinates
    /// @param[in] r_trg [ 3 x n_trg ] matrix of target coordinates
    /// @param[in] f_sl [ k_dim_sl x n_src ] matrix of 'single-layer' source strengths
    /// @param[in] f_dl [ k_dim_dl x n_src ] matrix of 'double-layer' source strengths
    /// @param[in] eta fluid viscosity
    /// @returns [ k_dim_trg x n_trg ] matrix of kernel evaluated at target positions given the sources
    Eigen::MatrixXd operator()(CMatrixRef &r_sl, CMatrixRef &r_dl, CMatrixRef &r_trg, CMatrixRef &f_sl,
                               CMatrixRef &f_dl, double eta) {
        // Check if source/target points have changed
        bool setup_flag =
            (force_device_sync_ || r_sl_old_.size() != r_sl.size() || r_dl_old_.size() != r_dl.size() ||
             r_trg_old_.size() != r_trg.size() || r_sl_old_ != r_sl || r_dl_old_ != r_dl || r_trg_old_ != r_trg);

        if (setup_flag) {
            // Update FMM tree and cache coordinates
            r_sl_old_ = r_sl;
            r_dl_old_ = r_dl;
            r_trg_old_ = r_trg;
            sync_device_positions();
            force_device_sync_ = false;
        }

        sync_device_forces(f_sl.data(), f_sl.size(), f_dl.data(), f_dl.size());
        return kernel_func_(r_sl, r_dl, r_trg, f_sl, f_dl, eta);
    }

    ~GPUEvaluator() { free_device_memory(); }

  private:
    void sync_device_positions();
    void sync_device_forces(const double *f_sl, size_t sl_size, const double *f_dl, size_t dl_size);
    void free_device_memory();

    bool force_device_sync_ = true; ///< When set, forces device re-transfer, then is cleared
    Eigen::MatrixXd r_sl_old_;      ///< cached 'single-layer' source positions to check for FMM tree invalidation
    Eigen::MatrixXd r_dl_old_;      ///< cache 'double-layer' source positions to check for FMM tree invalidation
    Eigen::MatrixXd r_trg_old_;     ///< cache target positions to check for FMM tree invalidation

    double *r_sl_device_;
    double *r_dl_device_;
    double *r_trg_device_;
    double *f_sl_device_;
    double *f_dl_device_;
    Evaluator kernel_func_; ///< Pointer to gpu pair evaluator
};
}; // namespace kernels

#endif
