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
    bool periphery_interaction_flag;
    bool adaptive_timestep_flag;

    periphery_binding_t periphery_binding{
        .polar_angle_start = 0.0,
        .polar_angle_end = M_PI,
        .threshold = 0.75,
    };

    struct {
        int n_nodes = 0;
        double v_growth;
        double f_catastrophe;
        double v_grow_collision_scale;
        double f_catastrophe_collision_scale;
        double nucleation_rate;
        double min_length;
        double radius;
        double bending_rigidity;
        double min_separation;
    } dynamic_instability;
    unsigned long seed;

    fiber_periphery_interaction_t fiber_periphery_interaction{
        .f_0 = 20.0,
        .l_0 = 0.05,
    };

    struct {
        bool moving_volume;
        double dt_write_field;
        double resolution;
        double moving_volume_radius;
    } velocity_field;

    struct {
        int body_stresslet_multipole_order = 8;
        int body_stresslet_max_points = 2000;
        int body_oseen_multipole_order = 8;
        int body_oseen_max_points = 2000;
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
