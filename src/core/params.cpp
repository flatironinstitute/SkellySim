#include <params.hpp>

Params::Params(toml::value &pt) {
    eta = toml::find_or(pt, "eta", 1.0);
    dt_initial = toml::find_or(pt, "dt_initial", 1E-2);
    gmres_tol = toml::find_or(pt, "gmres_tol", 1E-10);
    t_final = toml::find_or(pt, "t_final", 1.0);
    fiber_error_tol = toml::find_or(pt, "fiber_error_tol", 1E-1);
    dt_max = toml::find_or(pt, "dt_max", 2.0);
    beta_up = toml::find_or(pt, "beta_up", 1.2);
    beta_down = toml::find_or(pt, "beta_down", 0.5);
    dt_min = toml::find_or(pt, "dt_min", 1E-4);
    dt_write = toml::find_or(pt, "dt_write", 0.25);
    implicit_motor_activation_delay = toml::find_or(pt, "implicit_motor_activation_delay", 0.0);
    seed = toml::find_or(pt, "seed", 1);
    adaptive_timestep_flag = toml::find_or(pt, "adaptive_timestep_flag", true);
    pair_evaluator = toml::find_or(pt, "pair_evaluator", "FMM");

    if (pt.contains("STKFMM")) {
        const auto s = pt.at("STKFMM");
        stkfmm.fiber_stokeslet_multipole_order =
            toml::find_or(s, "fiber_stokeslet_multipole_order", stkfmm.fiber_stokeslet_multipole_order);
        stkfmm.fiber_stokeslet_max_points =
            toml::find_or(s, "fiber_stokeslet_oseen_max_points", stkfmm.fiber_stokeslet_max_points);
        stkfmm.periphery_stresslet_multipole_order =
            toml::find_or(s, "periphery_stresslet_multipole_order", stkfmm.periphery_stresslet_multipole_order);
        stkfmm.periphery_stresslet_max_points =
            toml::find_or(s, "periphery_stresslet_max_points", stkfmm.periphery_stresslet_max_points);
    }

}
