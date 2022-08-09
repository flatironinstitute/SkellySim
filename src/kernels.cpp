#include <kernels.hpp>

#include <STKFMM/STKFMM.hpp>

/*********************************************************
 *                                                        *
 *   Stokes Double Vel kernel, source: 9, target: 3       *
 *                                                        *
 **********************************************************/
struct stokes_doublevel : public pvfmm::GenericKernel<stokes_doublevel> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (8.0 * sctl::const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[3], const VecType (&r)[3], const VecType (&f)[9], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv3 = rinv * rinv * rinv;
        VecType rinv5 = rinv3 * rinv * rinv;

        const VecType sxx = f[0], sxy = f[1], sxz = f[2];
        const VecType syx = f[3], syy = f[4], syz = f[5];
        const VecType szx = f[6], szy = f[7], szz = f[8];
        const VecType dx = r[0], dy = r[1], dz = r[2];

        VecType commonCoeff = sxx * dx * dx + syy * dy * dy + szz * dz * dz;
        commonCoeff += (sxy + syx) * dx * dy;
        commonCoeff += (sxz + szx) * dx * dz;
        commonCoeff += (syz + szy) * dy * dz;
        commonCoeff *= (typename VecType::ScalarType)(-3.0) * rinv5;

        const VecType trace = sxx + syy + szz;
        u[0] += dx * commonCoeff;
        u[1] += dy * commonCoeff;
        u[2] += dz * commonCoeff;
    }
};

static auto g_memmgr = pvfmm::mem::MemoryManager(0);
Eigen::MatrixXd kernels::stokeslet_direct_cpu(MatrixRef &r_sl, MatrixRef &r_dl, MatrixRef &r_trg, MatrixRef &f_sl,
                                              MatrixRef &f_dl, double eta) {
    Eigen::MatrixXd u_trg = Eigen::MatrixXd::Zero(3, r_trg.cols());

    pvfmm::stokes_vel::Eval(const_cast<double *>(r_sl.data()), r_sl.cols(), const_cast<double *>(f_sl.data()), 1,
                            const_cast<double *>(r_trg.data()), r_trg.cols(), u_trg.data(), &g_memmgr);
    return u_trg / eta;
}

Eigen::MatrixXd kernels::stresslet_direct_cpu(MatrixRef &r_sl, MatrixRef &r_dl, MatrixRef &r_trg, MatrixRef &f_sl,
                                              MatrixRef &f_dl, double eta) {
    Eigen::MatrixXd u_trg = Eigen::MatrixXd::Zero(3, r_trg.cols());
    stokes_doublevel::Eval(const_cast<double *>(r_dl.data()), r_dl.cols(), const_cast<double *>(f_dl.data()), 1,
                           const_cast<double *>(r_trg.data()), r_trg.cols(), u_trg.data(), &g_memmgr);

    return u_trg / eta;
}

Eigen::MatrixXd kernels::oseen_tensor_contract_direct(MatrixRef &r_src, MatrixRef &r_trg, MatrixRef &density,
                                                      double eta, double reg, double epsilon_distance) {
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

/// Build the [ 3*n_src x 3*n_trg ] matrix of stokeslets 'Oseen tensor'
/// Terms where r_src = r_trg are left zero
///
/// G = f(r) * I + g(r) * (r.T*r)
///
/// Input:
///   r_vectors = coordinates.
///   eta = (default 1.0) viscosity
///   reg = (default 5e-3) regularization term
///   epsilon_distance = (default 1e-10) set elements to zero for distances < epsilon_distance.
///
/// Output:
///   G = Oseen tensor with dimensions (3*num_points) x (3*num_points).
Eigen::MatrixXd kernels::oseen_tensor_direct(MatrixRef &r_src, MatrixRef &r_trg, double eta, double reg,
                                             double epsilon_distance) {

    using Eigen::MatrixXd;
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

            if (dr2 == 0.0)
                continue;

            double dr = sqrt(dr2);

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

/// @brief Build the rotlet tensor for given source/target/density
///
/// @param[in] r_src [3 x n_src] matrix of source positions
/// @param[in] r_trg [3 x n_trg] matrix of target positions
/// @param[in] density [3 x n_src] matrix source strengths
/// @param[in] eta viscosity of fluid
/// @param[in] reg regularization term
/// @param[in] epsilon_distance threshold distance, below which source-target distances will be regularized
/// @return [3 x n_trg] rotlet at target given source points
Eigen::MatrixXd kernels::rotlet(MatrixRef &r_src, MatrixRef &r_trg, MatrixRef &density, double eta, double reg,
                                double epsilon_distance) {
    using Eigen::MatrixXd;
    const int N_src = r_src.size() / 3;
    const int N_trg = r_trg.size() / 3;
    const double eps2 = epsilon_distance * epsilon_distance;
    const double reg2 = reg * reg;
    const double factor = 1 / (8.0 * M_PI * eta);

    MatrixXd u = MatrixXd::Zero(3, r_trg.cols());

    // FIXME: Might want to vectorize rotlet calculation
    for (int i_trg = 0; i_trg < N_trg; ++i_trg) {
        for (int i_src = 0; i_src < N_src; ++i_src) {
            double dx = r_trg(0, i_trg) - r_src(0, i_src);
            double dy = r_trg(1, i_trg) - r_src(1, i_src);
            double dz = r_trg(2, i_trg) - r_src(2, i_src);
            double dr2 = dx * dx + dy * dy + dz * dz;

            double dr = dr2 < eps2 ? sqrt(reg2 + dr2) : sqrt(dr2);

            double fr = 1.0 / std::pow(dr, 3);
            double Mxy = fr * dz;
            double Mxz = -fr * dy;
            double Myx = -fr * dz;
            double Myz = fr * dx;
            double Mzx = fr * dy;
            double Mzy = -fr * dx;
            u(0, i_trg) += Mxy * density(1, i_src) + Mxz * density(2, i_src);
            u(1, i_trg) += Myx * density(0, i_src) + Myz * density(2, i_src);
            u(2, i_trg) += Mzx * density(0, i_src) + Mzy * density(1, i_src);
        }
    }
    u *= factor;

    return u;
}

/// Build the Stresslet tensor contracted with a vector for N points (sources and targets).
/// Set to zero diagonal terms.
///
/// S_ij = sum_k -(3/(4*pi)) * r_i * r_j * r_k
///
/// Input:
///    r_vectors = coordinates.
///    normal = vector used to contract the Stresslet (in general this will be the normal vector of a surface).
///    eta = (default 1.0) viscosity
///    reg = (default 5e-3) regularization term
///    epsilon_distance = (default 1e-10) set elements to zero for distances < epsilon_distance.
///
/// Output:
///    S_normal = Stresslet tensor contracted with a vector with dimensions (3*num_points) x (3*num_points).
/// Output with format
///               | S_normal11 S_normal12 ...|
///    S_normal = | S_normal21 S_normal22 ...|
///               | ...                      |
///     with S_normal12 the stresslet between points r_1 and r_2.
///     S_normal12 has dimensions 3 x 3.
Eigen::MatrixXd kernels::stresslet_times_normal(MatrixRef &r_src, MatrixRef &normals, double eta, double reg,
                                                double epsilon_distance) {
    const double factor = -3.0 / (4.0 * M_PI);
    const double reg2 = reg * reg;
    const int N = r_src.cols();
    Eigen::MatrixXd Snormal = Eigen::MatrixXd::Zero(3 * N, 3 * N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j)
                continue;

            const Eigen::Vector3d dr = r_src.col(i) - r_src.col(j);
            double r_norm = dr.norm();
            if (r_norm < epsilon_distance)
                r_norm = sqrt(r_norm * r_norm + reg2);

            const double r_inv5 = 1.0 / std::pow(r_norm, 5);
            Snormal.block(3 * i, 3 * j, 3, 3) = (factor * dr.dot(normals.col(j)) * r_inv5) * dr * dr.transpose();
        }
    }

    return Snormal;
}

/// @brief Build the Stresslet tensor contracted with two vectors for N points (sources and targets).
///
/// Set diagonal terms to zero.
///  \f[ {\bf d}_{i,j} = {\bf r}_{i} - {\bf r}_{j} \f]
///  \f[ {\bf S}_i = -\frac{3}{4\pi} \sum_{j \ne i}^N
///              \frac{\left( {\bf d}_{i,j} \cdot {\bf \rho}_j \right)
///              \left( {\bf d}_{i,j} \cdot {\bf n}_j \right)}
///              {\left|{\bf d}_{i,j}\right|^{5}} \f]
///
///   @param[in] r_src [3 x n_src] matrix of source positions
///   @param[in] normals [3 x n_src] matrix used to contract the Stresslet (in general this will be the normal vector of
///   a surface).
///   @param[in] density [3 x n_src] matrix used to contract the Stresslet (in general this will be a double layer
///   potential).
///   @param[in] eta viscosity of fluid
///   @param[in] reg regularization term
///   @param[in] epsilon_distance threshold distance, below which return elements will be zero
///   @return [3 x num_points] contracted stresslet tensor 'S_normal'
Eigen::MatrixXd kernels::stresslet_times_normal_times_density(MatrixRef &r_src, MatrixRef &normals, MatrixRef &density,
                                                              double eta, double reg, double epsilon_distance) {
    const int N = r_src.size() / 3;
    const double factor = -3.0 / (4.0 * M_PI);
    const double reg2 = reg * reg;
    Eigen::MatrixXd Sdn = Eigen::MatrixXd::Zero(3, r_src.cols());

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j)
                continue;

            const Eigen::Vector3d dr = r_src.col(i) - r_src.col(j);
            double r_norm = dr.norm();
            if (r_norm < epsilon_distance)
                r_norm = sqrt(r_norm * r_norm + reg2);

            const double r_inv5 = 1.0 / std::pow(r_norm, 5);
            const double f0 = dr.dot(density.col(j)) * dr.dot(normals.col(j)) * r_inv5;
            Sdn.col(i) += f0 * dr;
        }
    }

    Sdn *= factor;

    return Sdn;
}

// FIXME: FMM kernel functions are unnecessary. The kernel dimension and kernel enum value can
// be cached and just called directly from the FMM() operator.
Eigen::MatrixXd kernels::stokes_vel_fmm(const int n_trg, MatrixRef &f_sl, MatrixRef &f_dl, stkfmm::STKFMM *fmmPtr) {
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(3, n_trg);
    fmmPtr->clearFMM(stkfmm::KERNEL::Stokes);
    fmmPtr->evaluateFMM(stkfmm::KERNEL::Stokes, f_sl.size() / 3, f_sl.data(), n_trg, res.data(), f_dl.size() / 3,
                        f_dl.data());
    return res;
}

Eigen::MatrixXd kernels::stokes_pvel_fmm(const int n_trg, MatrixRef &f_sl, MatrixRef &f_dl, stkfmm::STKFMM *fmmPtr) {
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(4, n_trg);
    fmmPtr->clearFMM(stkfmm::KERNEL::PVel);
    fmmPtr->evaluateFMM(stkfmm::KERNEL::PVel, f_sl.size() / 4, f_sl.data(), n_trg, res.data(), f_dl.size() / 9,
                        f_dl.data());
    return res.block(1, 0, 3, n_trg);
}
