#include <skelly_sim.hpp>

#include "cnpy.hpp"
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <system.hpp>

#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#define NDEBUG
#else
#include <cassert>
#endif

#include <body.hpp>
#include <fiber.hpp>

template <typename DerivedA, typename DerivedB>
bool allclose(
    const Eigen::DenseBase<DerivedA> &a, const Eigen::DenseBase<DerivedB> &b,
    const typename DerivedA::RealScalar &rtol = Eigen::NumTraits<typename DerivedA::RealScalar>::dummy_precision(),
    const typename DerivedA::RealScalar &atol = Eigen::NumTraits<typename DerivedA::RealScalar>::epsilon()) {
    return ((a.derived() - b.derived()).array().abs() <= (atol + rtol * b.derived().array().abs())).all();
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    std::string config_file("test_body.toml");
    toml::value config = toml::parse(config_file);
    Params params(config.at("params"));
    toml::array &body_configs = config.at("bodies").as_array();
    Body body(body_configs.at(0).as_table(), params);

    System::init(config_file);
    FiberContainer &fc = *System::get_fiber_container();
    BodyContainer &bc = *System::get_body_container();
    for (auto &fiber : fc.fibers) {
        auto [i_body, i_site] = fiber.binding_site_;
        auto &body = *bc.bodies[i_body];

        if (i_site > 0)
            std::cout << i_body << " " << i_site << " [" << body.nucleation_sites_ref_.col(i_site).transpose() << "] ["
                      << fiber.x_.col(0).transpose() << "]" << std::endl;
    }

    MPI_Finalize();

    std::cout << "Test passed\n";
    return 0;
}
