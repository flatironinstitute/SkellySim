#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <fstream>
#include <skelly_sim.hpp>

class Params;
class BodyContainer;
class FiberContainerBase;
class Periphery;
class PointSourceContainer;

/// Namespace for System, which drives the simulation and handles communication (timestepping, data wrangling, etc)
namespace System {

/// @brief Time varying system properties that are extrinsic to the physical objects
struct properties_t {
    double dt;         ///< Current timestep size
    double time = 0.0; ///< Current system time
};

void init(const std::string &input_file, bool resume_flag = false, bool listen_flag = false);
Params *get_params();
BodyContainer *get_body_container();
FiberContainerBase *get_fiber_container();
Periphery *get_shell();
PointSourceContainer *get_point_source_container();
toml::value *get_param_table();

Eigen::MatrixXd calculate_body_fiber_link_conditions(VectorRef &fibers_xt, VectorRef &x_bodies);
std::tuple<int, int, int> get_local_solution_sizes();
Eigen::VectorXd apply_preconditioner(VectorRef &x);
Eigen::VectorXd apply_matvec(VectorRef &x);
void dynamic_instability();
void prep_state_for_solver();
bool solve();
bool step();
void run();
void write();
void write(std::ofstream &);
void write_header(std::ofstream &);
bool check_collision();
void backup();
void restore();
void set_evaluator(const std::string &evaluator);

Eigen::VectorXd get_fiber_RHS();
Eigen::VectorXd get_shell_RHS();
Eigen::VectorXd get_body_RHS();
struct properties_t &get_properties();
Eigen::VectorXd &get_curr_solution();
std::tuple<VectorMap, VectorMap, VectorMap> get_solution_maps(double *x);
std::tuple<CVectorMap, CVectorMap, CVectorMap> get_solution_maps(const double *x);
Eigen::MatrixXd velocity_at_targets(MatrixRef &r_trg);

}; // namespace System

#endif
