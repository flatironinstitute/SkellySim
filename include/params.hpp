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
    double tol_tstep;
    double dt_min;
    double dt_max;
    struct {
        int n_nodes = 0;
        double v_growth;
        double f_catastrophe;
        double v_grow_collision_scale;
        double f_catastrophe_collision_scale;
        double nucleation_rate;
        double min_length;
        double bending_rigidity;
    } dynamic_instability;
    unsigned long seed;

    std::string shell_precompute_file;

    Params(){};
    Params(toml::value &param_table);
};

#endif
