#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <body.hpp>
#include <fiber.hpp>
#include <params.hpp>
#include <periphery.hpp>

class System {
    Params params_;
    FiberContainer fc_;
    BodyContainer bc_;
    Periphery shell_;

    static System &get_instance_impl(std::string *const input_file = nullptr) {
        static System instance(input_file);
        return instance;
    };

    System(std::string *input_file = nullptr);

  public:
    static System &get_instance() { return get_instance_impl(); }
    static void init(std::string input_file) { get_instance_impl(&input_file); }
    static Params &get_params() { return get_instance_impl().params_; };
    static FiberContainer &get_fiber_container() { return get_instance_impl().fc_; };
    static BodyContainer &get_body_container() { return get_instance_impl().bc_; };
    static Periphery &get_shell() { return get_instance_impl().shell_; };
    static std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
    calculate_body_fiber_link_conditions(const FiberContainer &fc, const BodyContainer &bc,
                                         const Eigen::Ref<const Eigen::VectorXd> &fibers_xt,
                                         const Eigen::Ref<const Eigen::MatrixXd> &body_velocities);
};

#endif
