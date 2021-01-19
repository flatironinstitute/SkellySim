#include <params.hpp>

Params::Params(toml::table *param_table) {
    toml::table &pt = *param_table;
    eta = pt["eta"].value_or(1.0);
    dt = pt["dt"].value_or(0.005);
    gmres_tol = pt["gmres_tol"].value_or(1E-12);

    shell_precompute_file = pt["shell_precompute_file"].value_or("");
}
