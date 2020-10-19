#include "cnpy.hpp"
#include <Eigen/Core>
#include <iostream>
#include <mpi.h>

#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#define NDEBUG
#else
#include <cassert>
#endif

#include <fiber.hpp>

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd load_mat(cnpy::npz_t &npz, const char *var) {
    return Map<Eigen::ArrayXXd>(npz[var].data<double>(), npz[var].shape[1], npz[var].shape[0]).matrix();
}

VectorXd load_vec(cnpy::npz_t &npz, const char *var) {
    return Map<Eigen::VectorXd>(npz[var].data<double>(), npz[var].shape[0]);
}

template <typename DerivedA, typename DerivedB>
bool allclose(
    const Eigen::DenseBase<DerivedA> &a, const Eigen::DenseBase<DerivedB> &b,
    const typename DerivedA::RealScalar &rtol = Eigen::NumTraits<typename DerivedA::RealScalar>::dummy_precision(),
    const typename DerivedA::RealScalar &atol = Eigen::NumTraits<typename DerivedA::RealScalar>::epsilon()) {
    return ((a.derived() - b.derived()).array().abs() <= (atol + rtol * b.derived().array().abs())).all();
}

int main(int argc, char *argv[]) {
    cnpy::npz_t np_fib = cnpy::npz_load("np_fib.npz");

    MatrixXd x = load_mat(np_fib, "x");
    MatrixXd xs = load_mat(np_fib, "xs");
    MatrixXd xss = load_mat(np_fib, "xss");
    MatrixXd xsss = load_mat(np_fib, "xsss");
    MatrixXd xssss = load_mat(np_fib, "xssss");

    MatrixXd D_1_0 = load_mat(np_fib, "D_1_0");
    MatrixXd D_2_0 = load_mat(np_fib, "D_2_0");
    MatrixXd D_3_0 = load_mat(np_fib, "D_3_0");
    MatrixXd D_4_0 = load_mat(np_fib, "D_4_0");

    MatrixXd force_external = load_mat(np_fib, "force_external");
    MatrixXd flow_on = load_mat(np_fib, "flow_on");

    MatrixXd A = load_mat(np_fib, "A").transpose();
    VectorXd RHS = load_vec(np_fib, "RHS");
    auto &mat = Fiber::matrices_.at(x.cols());

    assert(allclose(mat.D_1_0, D_1_0, 0, 1E-9));
    assert(allclose(mat.D_2_0, D_2_0, 0, 1E-9));
    assert(allclose(mat.D_3_0, D_3_0, 0, 1E-9));
    assert(allclose(mat.D_4_0, D_4_0, 0, 1E-9));

    double bending_rigidity = *np_fib["bending_rigidity"].data<double>();
    double length = *np_fib["length"].data<double>();
    double eta = *np_fib["eta"].data<double>();
    double dt = *np_fib["dt"].data<double>();

    Fiber fib(x.cols(), bending_rigidity, eta);
    fib.length_ = length;
    fib.x_ = x;
    fib.update_derivatives();

    assert(allclose(fib.x_, x, 0, 1E-9));
    assert(allclose(fib.xs_, xs, 0, 1E-9));
    assert(allclose(fib.xss_, xss, 0, 1E-9));
    assert(allclose(fib.xsss_, xsss, 0, 1E-9));
    assert(allclose(fib.xssss_, xssss, 0, 1E-9));

    fib.update_stokeslet(eta);
    MatrixXd stokeslet = load_mat(np_fib, "stokeslet");
    assert(allclose(fib.stokeslet_, stokeslet));

    fib.form_linear_operator(dt, eta);
    int np = fib.num_points_;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            Eigen::MatrixXd Apy = A.block(i * np, j * np, np, np);
            Eigen::MatrixXd Acpp = fib.A_.block(i * np, j * np, np, np);
            assert(allclose(Apy, Acpp, 0, 1E-7));
        }
    }

    // Test right-hand-side computation
    fib.compute_RHS(dt, flow_on, force_external);
    assert(allclose(RHS, fib.RHS_));

    // Check translation operator
    Eigen::Vector3d pt1 = fib.x_.col(1);
    fib.translate({1.0, 0.0, 0.0});
    assert(fib.x_(0, 1) == pt1(0) + 1.0);
    assert(fib.x_(1, 1) == pt1(1));
    assert(fib.x_(2, 1) == pt1(2));

    std::cout << "Test passed\n";
    return 0;
}
