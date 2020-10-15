#include <kernels.hpp>

Eigen::MatrixX3d kernels::oseen_tensor_contract_direct(const Eigen::Ref<Eigen::MatrixXd> &r_src,
                                                       const Eigen::Ref<Eigen::MatrixXd> &r_trg,
                                                       const Eigen::Ref<Eigen::MatrixXd> &density, double eta,
                                                       double reg, double epsilon_distance) {

    using namespace Eigen;
    const int N_src = r_src.size() / 3;
    const int N_trg = r_trg.size() / 3;

    // # Compute matrix of size 3N \times 3N
    MatrixX3d res = MatrixXd::Zero(N_trg, 3);

    const double factor = 1.0 / (8.0 * M_PI * eta);
    const double reg2 = std::pow(reg, 2);
    for (int i_trg = 0; i_trg < N_trg; ++i_trg) {
        for (int i_src = 0; i_src < N_src; ++i_src) {
            double fr, gr;
            double dx = r_src(i_src, 0) - r_trg(i_trg, 0);
            double dy = r_src(i_src, 1) - r_trg(i_trg, 1);
            double dz = r_src(i_src, 2) - r_trg(i_trg, 2);
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

            res(i_trg, 0) += Mxx * density(i_src, 0) + Mxy * density(i_src, 1) + Mxz * density(i_src, 2);
            res(i_trg, 1) += Mxy * density(i_src, 0) + Myy * density(i_src, 1) + Myz * density(i_src, 2);
            res(i_trg, 2) += Mxz * density(i_src, 0) + Myz * density(i_src, 1) + Mzz * density(i_src, 2);
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
            double dx = r_src(i_src, 0) - r_trg(i_trg, 0);
            double dy = r_src(i_src, 1) - r_trg(i_trg, 1);
            double dz = r_src(i_src, 2) - r_trg(i_trg, 2);
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
