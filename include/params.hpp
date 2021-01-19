#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <toml.hpp>

class Params {
  public:
    double eta;
    double dt;
    double gmres_tol;

    std::string shell_precompute_file;

    Params() {};
    Params(toml::table *param_table);
};

#endif
