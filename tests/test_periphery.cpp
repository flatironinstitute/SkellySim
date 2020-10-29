#include "cnpy.hpp"
#include <Eigen/Core>
#include <iostream>

#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#define NDEBUG
#else
#include <cassert>
#endif

#include <periphery.hpp>

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

    SphericalPeriphery(6, 5.0);

    std::cout << "Test passed\n";
    return 0;
}
