#include <params.hpp>

Params::Params(toml::value &pt) {
    eta = toml::find_or(pt, "eta", 1.0);
    dt_initial = toml::find_or(pt, "dt_initial", 1E-2);
    gmres_tol = toml::find_or(pt, "gmres_tol", 1E-2);
    t_final = toml::find_or(pt, "t_final", 1.0);
    tol_tstep = toml::find_or(pt, "tol_tstep", 1E-2);
    dt_max = toml::find_or(pt, "dt_max", 2.0);
    beta_up = toml::find_or(pt, "beta_up", 1.2);
    beta_down = toml::find_or(pt, "beta_down", 0.5);
    seed = toml::find_or(pt, "seed", 1);
    dt_min = toml::find_or(pt, "dt_min", 1E-4);

    shell_precompute_file = toml::find_or(pt, "shell_precompute_file", "");
}
