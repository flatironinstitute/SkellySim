#include <params.hpp>

Params::Params(toml::table *param_table) {
    toml::table &pt = *param_table;
    eta = pt["eta"].value_or(1.0);
    dt_initial = pt["dt_initial"].value_or(1E-2);
    gmres_tol = pt["gmres_tol"].value_or(1E-12);
    t_final = pt["t_final"].value_or(1.0);
    tol_tstep = pt["tol_tstep"].value_or(1E-2);
    dt_max = pt["dt_max"].value_or(2.0);
    beta_up = pt["beta_up"].value_or(1.2);
    beta_down = pt["beta_down"].value_or(0.5);
    seed = pt["seed"].value_or(1);
    dt_min = pt["dt_min"].value_or(1E-4);

    shell_precompute_file = pt["shell_precompute_file"].value_or("");
}
