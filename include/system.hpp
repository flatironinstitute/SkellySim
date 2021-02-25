#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <skelly_sim.hpp>

#include <body.hpp>
#include <fiber.hpp>
#include <params.hpp>
#include <periphery.hpp>

class System {
    Params params_;
    FiberContainer fc_;
    BodyContainer bc_;
    Periphery shell_;
    toml::table param_table_;

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
    static toml::table &get_param_table() { return get_instance_impl().param_table_; };
    static std::pair<Eigen::MatrixXd, Eigen::MatrixXd> calculate_body_fiber_link_conditions(VectorRef &fibers_xt,
                                                                                            MatrixRef &body_velocities);
    static std::tuple<int, int, int> get_local_solution_sizes();
    static Eigen::VectorXd apply_preconditioner(VectorRef &x);
    static Eigen::VectorXd apply_matvec(VectorRef &x);
};

#endif
