#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <fstream>
#include <skelly_sim.hpp>

class Params;
class BodyContainer;
class FiberContainer;
class Periphery;

/// Namespace for System, which drives the simulation and handles communication (timestepping, data wrangling, etc)
namespace System {

/// @brief Time varying system properties that are extrinsic to the physical objects
struct properties_t {
    double dt;         ///< Current timestep size
    double time = 0.0; ///< Current system time
};

void init(const std::string &input_file, bool resume_flag = false, bool post_process_flag = false);
Params *get_params();
BodyContainer *get_body_container();
FiberContainer *get_fiber_container();
Periphery *get_shell();
toml::value *get_param_table();

Eigen::MatrixXd calculate_body_fiber_link_conditions(VectorRef &fibers_xt, VectorRef &x_bodies);
std::tuple<int, int, int> get_local_solution_sizes();
Eigen::VectorXd apply_preconditioner(VectorRef &x);
Eigen::VectorXd apply_matvec(VectorRef &x);
void dynamic_instability();
void prep_state_for_solver();
bool step();
void run();
void run_post_process();
void write();
void write(std::ofstream &);
bool check_collision();
void backup();
void restore();
Eigen::VectorXd get_fiber_RHS();
Eigen::VectorXd get_shell_RHS();
Eigen::VectorXd get_body_RHS();
struct properties_t &get_properties();

}; // namespace System

#endif
