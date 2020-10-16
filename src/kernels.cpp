#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <kernels.hpp>

#include <STKFMM/STKFMM.hpp>

Eigen::MatrixXd kernels::oseen_tensor_contract_direct(const Eigen::Ref<Eigen::MatrixXd> &r_src,
                                                      const Eigen::Ref<Eigen::MatrixXd> &r_trg,
                                                      const Eigen::Ref<Eigen::MatrixXd> &density, double eta,
                                                      double reg, double epsilon_distance) {

    using namespace Eigen;
    const int N_src = r_src.size() / 3;
    const int N_trg = r_trg.size() / 3;

    // # Compute matrix of size 3N \times 3N
    MatrixXd res = MatrixXd::Zero(3, N_trg);

    const double factor = 1.0 / (8.0 * M_PI * eta);
    const double reg2 = std::pow(reg, 2);
    for (int i_trg = 0; i_trg < N_trg; ++i_trg) {
        for (int i_src = 0; i_src < N_src; ++i_src) {
            double fr, gr;
            double dx = r_src(0, i_src) - r_trg(0, i_trg);
            double dy = r_src(1, i_src) - r_trg(1, i_trg);
            double dz = r_src(2, i_src) - r_trg(2, i_trg);
            double dr2 = dx * dx + dy * dy + dz * dz;
            double dr = sqrt(dr2);

            if (dr == 0.0)
                continue;

            if (dr > epsilon_distance) {
                fr = factor / dr;
                gr = factor / std::pow(dr, 3);
            } else {
                double denom_inv = 1.0 / sqrt(std::pow(dr, 2) + reg2);
                fr = factor * denom_inv;
                gr = factor * std::pow(denom_inv, 3);
            }

            double Mxx = fr + gr * dx * dx;
            double Mxy = gr * dx * dy;
            double Mxz = gr * dx * dz;
            double Myy = fr + gr * dy * dy;
            double Myz = gr * dy * dz;
            double Mzz = fr + gr * dz * dz;

            res(0, i_trg) += Mxx * density(0, i_src) + Mxy * density(1, i_src) + Mxz * density(2, i_src);
            res(1, i_trg) += Mxy * density(0, i_src) + Myy * density(1, i_src) + Myz * density(2, i_src);
            res(2, i_trg) += Mxz * density(0, i_src) + Myz * density(1, i_src) + Mzz * density(2, i_src);
        }
    }

    return res;
}

// Build the Oseen tensor for N points (sources == targets).
// Set to zero diagonal terms.
//
// G = f(r) * I + g(r) * (r.T*r)
//
// Input:
//   r_vectors = coordinates.
//   eta = (default 1.0) viscosity
//   reg = (default 5e-3) regularization term
//   epsilon_distance = (default 1e-10) set elements to zero for distances < epsilon_distance.
//
// Output:
//   G = Oseen tensor with dimensions (3*num_points) x (3*num_points).
Eigen::MatrixXd kernels::oseen_tensor_direct(const Eigen::Ref<Eigen::MatrixXd> &r_src,
                                             const Eigen::Ref<Eigen::MatrixXd> &r_trg, double eta, double reg,
                                             double epsilon_distance) {

    using namespace Eigen;
    const int N_src = r_src.size() / 3;
    const int N_trg = r_trg.size() / 3;

    // # Compute matrix of size 3N \times 3N
    MatrixXd G = MatrixXd::Zero(r_trg.size(), r_src.size());

    const double factor = 1.0 / (8.0 * M_PI * eta);
    const double reg2 = std::pow(reg, 2);
    for (int i_src = 0; i_src < N_trg; ++i_src) {
        for (int i_trg = 0; i_trg < N_src; ++i_trg) {
            double fr, gr;
            double dx = r_src(0, i_src) - r_trg(0, i_trg);
            double dy = r_src(1, i_src) - r_trg(1, i_trg);
            double dz = r_src(2, i_src) - r_trg(2, i_trg);
            double dr2 = dx * dx + dy * dy + dz * dz;
            double dr = sqrt(dr2);

            if (dr == 0.0)
                continue;

            if (dr > epsilon_distance) {
                fr = factor / dr;
                gr = factor / std::pow(dr, 3);
            } else {
                double denom_inv = 1.0 / sqrt(std::pow(dr, 2) + reg2);
                fr = factor * denom_inv;
                gr = factor * std::pow(denom_inv, 3);
            }

            G(i_trg * 3 + 0, i_src * 3 + 0) = fr + gr * dx * dx;
            G(i_trg * 3 + 0, i_src * 3 + 1) = gr * dx * dy;
            G(i_trg * 3 + 0, i_src * 3 + 2) = gr * dx * dz;

            G(i_trg * 3 + 1, i_src * 3 + 0) = gr * dy * dx;
            G(i_trg * 3 + 1, i_src * 3 + 1) = fr + gr * dy * dy;
            G(i_trg * 3 + 1, i_src * 3 + 2) = gr * dy * dz;

            G(i_trg * 3 + 2, i_src * 3 + 0) = gr * dz * dx;
            G(i_trg * 3 + 2, i_src * 3 + 1) = gr * dz * dy;
            G(i_trg * 3 + 2, i_src * 3 + 2) = fr + gr * dz * dz;
        }
    }

    return G;
}

Eigen::MatrixXd kernels::oseen_tensor_contract_fmm(const Eigen::Ref<Eigen::MatrixXd> &r_src,
                                                   const Eigen::Ref<Eigen::MatrixXd> &r_trg,
                                                   const Eigen::Ref<Eigen::MatrixXd> &f_src) {
    Teuchos::RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank();
    const int nprocs = comm->getSize();
    const unsigned k = static_cast<unsigned>(stkfmm::KERNEL::Stokes);
    const stkfmm::PAXIS paxis = static_cast<stkfmm::PAXIS>(0);
    const int maxPoints = 50;

    Teuchos::RCP<stkfmm::STKFMM> fmmPtr_ = Teuchos::rcp(new stkfmm::Stk3DFMM(8, maxPoints, paxis, k));
    fmmPtr_->showActiveKernels();
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(3, r_trg.size() / 3);

    double origin[3] = {0.0};
    fmmPtr_->setBox(origin, 1.0);

    const int n_trg_local = r_trg.size() / 3;

    fmmPtr_->setPoints(r_src.size() / 3, r_src.data(), n_trg_local, r_trg.data());

    fmmPtr_->setupTree(stkfmm::KERNEL::Stokes);
    fmmPtr_->evaluateFMM(stkfmm::KERNEL::Stokes, f_src.size() / 3, f_src.data(), n_trg_local, res.data());

    return res;
}
