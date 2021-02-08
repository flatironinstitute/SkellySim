#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <toml.hpp>

/// Class containing input parameters for the simulated system
class Params {
  public:
    double eta;
    double dt;
    double gmres_tol;
    unsigned long seed;

    std::string shell_precompute_file;

    Params() {};
    Params(toml::table *param_table);
};

#endif
