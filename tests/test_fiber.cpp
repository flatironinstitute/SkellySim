#include "cnpy.hpp"
#include <Eigen/Core>
#include <assert.h>
#include <iostream>
#include <mpi.h>

#include <fiber.hpp>

using Eigen::Map;
using Eigen::MatrixXd;

MatrixXd load_mat(cnpy::npz_t &npz, const char *var) {
    return Map<Eigen::ArrayXXd>(npz[var].data<double>(), npz[var].shape[1], npz[var].shape[0]).matrix().transpose();
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

    auto &mat = Fiber::matrices.at(x.rows());

    assert(allclose(mat.D_1_0, D_1_0));
    assert(allclose(mat.D_2_0, D_2_0));
    assert(allclose(mat.D_3_0, D_3_0));
    assert(allclose(mat.D_4_0, D_4_0));

    double bending_rigidity = *np_fib["bending_rigidity"].data<double>();
    double length = *np_fib["length"].data<double>();
    double eta = *np_fib["eta"].data<double>();

    Fiber fib(x.rows(), bending_rigidity);
    fib.length = length;
    fib.x = x;
    fib.update_derivatives();

    assert(allclose(fib.x, x, 0, 1E-9));
    assert(allclose(fib.xs, xs, 0, 1E-9));
    assert(allclose(fib.xss, xss, 0, 1E-9));
    assert(allclose(fib.xsss, xsss, 0, 1E-9));
    assert(allclose(fib.xssss, xssss, 0, 1E-9));

    fib.update_stokeslet(eta);
    MatrixXd stokeslet = load_mat(np_fib, "stokeslet");
    assert(allclose(fib.stokeslet, stokeslet));

    std::cout << "Test passed\n";
    return 0;
}
