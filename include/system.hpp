#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <skelly_sim.hpp>

class Params;
class BodyContainer;
class FiberContainer;
class Periphery;

namespace System {
void backup_impl();
void restore_impl();

void init(const std::string &input_file);
Params *get_params();
BodyContainer *get_body_container();
FiberContainer *get_fiber_container();
Periphery *get_shell();
toml::value *get_param_table();

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> calculate_body_fiber_link_conditions(VectorRef &fibers_xt,
                                                                                 MatrixRef &body_velocities);
std::tuple<int, int, int> get_local_solution_sizes();
Eigen::VectorXd apply_preconditioner(VectorRef &x);
Eigen::VectorXd apply_matvec(VectorRef &x);
void dynamic_instability();
bool step();
void run();
bool check_collision();
void backup();
void restore();
Eigen::VectorXd get_fiber_RHS();
Eigen::VectorXd get_shell_RHS();
Eigen::VectorXd get_body_RHS();

}; // namespace System

#endif
