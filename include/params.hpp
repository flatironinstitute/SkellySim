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
    unsigned long seed;

    std::string shell_precompute_file;

    Params() {};
    Params(toml::table *param_table);
};

#endif
