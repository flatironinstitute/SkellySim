#ifndef KERNELS_HPP
#define KERNELS_HPP
#include <Eigen/Dense>
#include <STKFMM/STKFMM.hpp>


namespace kernels {
typedef Eigen::MatrixXd (*fmm_kernel_func_t)(const Eigen::Ref<Eigen::MatrixXd> &, const Eigen::Ref<Eigen::MatrixXd> &,
                                             const Eigen::Ref<Eigen::MatrixXd> &, stkfmm::STKFMM *, stkfmm::KERNEL);

Eigen::MatrixXd oseen_tensor_contract_direct(const Eigen::Ref<Eigen::MatrixXd> &r_src,
                                             const Eigen::Ref<Eigen::MatrixXd> &r_trg,
                                             const Eigen::Ref<Eigen::MatrixXd> &density, double eta = 1.0,
                                             double reg = 5E-3, double epsilon_distance = 1E-5);

Eigen::MatrixXd oseen_tensor_contract_fmm(const Eigen::Ref<Eigen::MatrixXd> &r_src,
                                          const Eigen::Ref<Eigen::MatrixXd> &r_trg,
                                          const Eigen::Ref<Eigen::MatrixXd> &f_src, stkfmm::STKFMM *fmmPtr,
                                          stkfmm::KERNEL k);

Eigen::MatrixXd oseen_tensor_direct(const Eigen::Ref<Eigen::MatrixXd> &r_src, const Eigen::Ref<Eigen::MatrixXd> &r_trg,
                                    double eta = 1.0, double reg = 5E-3, double epsilon_distance = 1E-5);

template <typename stkfmm_type>
class FMM {
  public:
    template <typename F>
    FMM(const int order, const int maxPoints, const stkfmm::PAXIS paxis, const stkfmm::KERNEL k, const F &kernel_func)
        : fmmPtr_(new stkfmm_type(order, maxPoints, paxis, static_cast<unsigned>(k))), k_(k),
          kernel_func_(kernel_func){};

    Eigen::MatrixXd operator()(const Eigen::Ref<Eigen::MatrixXd> &r_src, const Eigen::Ref<Eigen::MatrixXd> &r_trg,
                               const Eigen::Ref<Eigen::MatrixXd> &f_src) {
        if (r_src_old_.size() != r_src.size() || r_trg_old_.size() != r_trg.size() || r_src_old_ != r_src ||
            r_trg_old_ != r_trg) {
            // FIXME: Origin/size shouldn't be so fixed
            double origin[3] = {0.0};
            fmmPtr_->setBox(origin, 100.);

            fmmPtr_->setPoints(r_src.size() / 3, r_src.data(), r_trg.size() / 3, r_trg.data());
            fmmPtr_->setupTree(stkfmm::KERNEL::Stokes);
            r_src_old_ = r_src;
            r_trg_old_ = r_trg;
        }

        return kernel_func_(r_src, r_trg, f_src, fmmPtr_.get(), k_);
    }

  private:
    std::unique_ptr<stkfmm::STKFMM> fmmPtr_;
    stkfmm::KERNEL k_;
    Eigen::MatrixXd r_src_old_;
    Eigen::MatrixXd r_trg_old_;
    fmm_kernel_func_t kernel_func_;
};
}; // namespace kernels

#endif
