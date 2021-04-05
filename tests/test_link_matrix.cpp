#include "cnpy.hpp"
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <parse_util.hpp>
#include <system.hpp>

#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#define NDEBUG
#else
#include <cassert>
#endif

#include <body.hpp>

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;

template <typename DerivedA, typename DerivedB>
bool allclose(
    const Eigen::DenseBase<DerivedA> &a, const Eigen::DenseBase<DerivedB> &b,
    const typename DerivedA::RealScalar &rtol = Eigen::NumTraits<typename DerivedA::RealScalar>::dummy_precision(),
    const typename DerivedA::RealScalar &atol = Eigen::NumTraits<typename DerivedA::RealScalar>::epsilon()) {
    return ((a.derived() - b.derived()).array().abs() <= (atol + rtol * b.derived().array().abs())).all();
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    std::string config_file("test_link_matrix.toml");
    System::init(config_file);
    FiberContainer &fc = System::get_fiber_container();
    toml::value &param_table = System::get_param_table();

    toml::table special = param_table["special"].as_table();
    VectorXd fibers_xt = parse_util::convert_array(special.at("fibers_xt").as_array());
    VectorXd body_velocities_flat = parse_util::convert_array(special.at("body_velocities").as_array());
    VectorXd force_torque_ref = parse_util::convert_array(special.at("force_torque").as_array());
    VectorXd velocities_on_fiber_ref = parse_util::convert_array(special.at("fiber_velocities").as_array());
    MatrixXd body_velocities = Map<MatrixXd>(body_velocities_flat.data(), 6, body_velocities_flat.size() / 6);

    fc.update_derivatives();

    MatrixXd force_torque, velocities_on_fiber;
    std::tie(force_torque, velocities_on_fiber) =
        System::calculate_body_fiber_link_conditions(fibers_xt, body_velocities);

    assert(allclose(force_torque, force_torque_ref, 1E-7));
    assert(allclose(velocities_on_fiber, velocities_on_fiber_ref));
    MPI_Finalize();

    std::cout << "Test passed\n";
    return 0;
}
