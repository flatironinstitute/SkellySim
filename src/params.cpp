#include <params.hpp>

Params::Params(toml::table *param_table) {
    eta = param_table->get("eta")->value_or(1.0);
    dt = param_table->get("dt")->value_or(0.005);
    gmres_tol = param_table->get("gmres_tol")->value_or(1E-12);

    shell_precompute_file = param_table->get("shell_precompute_file")->value_or("");
    body_precompute_file = param_table->get("body_precompute_file")->value_or("");
}
