#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <skelly_sim.hpp>

/// Class containing input parameters for the simulated system
class Params {
  public:
    double eta;
    double dt_initial;
    double beta_up;
    double beta_down;
    double gmres_tol;
    double t_final;
    double fiber_error_tol;
    double dt_min;
    double dt_max;
    double dt_write;
    double implicit_motor_activation_delay;
    bool adaptive_timestep_flag;
    std::string pair_evaluator;
    unsigned long seed;

    struct {
        int fiber_stokeslet_multipole_order = 8;
        int fiber_stokeslet_max_points = 2000;
        int periphery_stresslet_multipole_order = 8;
        int periphery_stresslet_max_points = 2000;
    } stkfmm;

    std::string shell_precompute_file;

    Params() = default;
    Params(toml::value &param_table);
};

#endif
