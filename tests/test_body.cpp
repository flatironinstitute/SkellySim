#include "cnpy.hpp"
#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <mpi.h>

#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#define NDEBUG
#else
#include <cassert>
#endif

#include <body.hpp>

template <typename DerivedA, typename DerivedB>
bool allclose(
    const Eigen::DenseBase<DerivedA> &a, const Eigen::DenseBase<DerivedB> &b,
    const typename DerivedA::RealScalar &rtol = Eigen::NumTraits<typename DerivedA::RealScalar>::dummy_precision(),
    const typename DerivedA::RealScalar &atol = Eigen::NumTraits<typename DerivedA::RealScalar>::epsilon()) {
    return ((a.derived() - b.derived()).array().abs() <= (atol + rtol * b.derived().array().abs())).all();
}

int main(int argc, char *argv[]) {
    toml::table config = toml::parse_file("test_body.toml");
    Params params(config.get_as<toml::table>("params"));
    toml::array *body_configs = config.get_as<toml::array>("bodies");
    Body body(body_configs->get_as<toml::table>(0), params);

    std::cout << "Test passed\n";
    return 0;
}
