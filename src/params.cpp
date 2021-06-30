#include <params.hpp>

#include <mpi.h>

Params::Params(toml::value &pt) {
    eta = toml::find_or(pt, "eta", 1.0);
    dt_initial = toml::find_or(pt, "dt_initial", 1E-2);
    gmres_tol = toml::find_or(pt, "gmres_tol", 1E-10);
    t_final = toml::find_or(pt, "t_final", 1.0);
    fiber_error_tol = toml::find_or(pt, "fiber_error_tol", 1E-1);
    dt_max = toml::find_or(pt, "dt_max", 2.0);
    beta_up = toml::find_or(pt, "beta_up", 1.2);
    beta_down = toml::find_or(pt, "beta_down", 0.5);
    seed = toml::find_or(pt, "seed", 1);
    dt_min = toml::find_or(pt, "dt_min", 1E-4);
    dt_write = toml::find_or(pt, "dt_write", 0.25);
    seed = toml::find_or(pt, "seed", 1);
    periphery_binding_flag = toml::find_or(pt, "periphery_binding_flag", false);

    if (pt.contains("dynamic_instability")) {
        const auto &di = pt.at("dynamic_instability");
        try {
            dynamic_instability.n_nodes = toml::find(di, "n_nodes").as_integer();
            dynamic_instability.v_growth = toml::find(di, "v_growth").as_floating();
            dynamic_instability.f_catastrophe = toml::find(di, "f_catastrophe").as_floating();
            dynamic_instability.v_grow_collision_scale = toml::find(di, "v_grow_collision_scale").as_floating();
            dynamic_instability.f_catastrophe_collision_scale =
                toml::find(di, "f_catastrophe_collision_scale").as_floating();
            dynamic_instability.nucleation_rate = toml::find(di, "nucleation_rate").as_floating();
            dynamic_instability.min_length = toml::find(di, "min_length").as_floating();
            dynamic_instability.bending_rigidity = toml::find(di, "bending_rigidity").as_floating();
            dynamic_instability.min_separation = toml::find(di, "min_separation").as_floating();
        } catch (std::runtime_error &e) {
            MPI_Finalize();
            exit(1);
        }
    }

    if (pt.contains("STKFMM")) {
        const auto s = pt.at("STKFMM");
        stkfmm.body_stresslet_multipole_order =
            toml::find_or(s, "body_stresslet_multipole_order", stkfmm.body_stresslet_multipole_order);
        stkfmm.body_stresslet_max_points =
            toml::find_or(s, "body_stresslet_max_points", stkfmm.body_stresslet_max_points);
        stkfmm.body_oseen_multipole_order =
            toml::find_or(s, "body_oseen_multipole_order", stkfmm.body_oseen_multipole_order);
        stkfmm.body_oseen_max_points = toml::find_or(s, "body_oseen_max_points", stkfmm.body_oseen_max_points);
        stkfmm.fiber_stokeslet_multipole_order =
            toml::find_or(s, "fiber_stokeslet_multipole_order", stkfmm.fiber_stokeslet_multipole_order);
        stkfmm.fiber_stokeslet_max_points =
            toml::find_or(s, "fiber_stokeslet_oseen_max_points", stkfmm.fiber_stokeslet_max_points);
        stkfmm.periphery_stresslet_multipole_order =
            toml::find_or(s, "periphery_stresslet_multipole_order", stkfmm.periphery_stresslet_multipole_order);
        stkfmm.periphery_stresslet_max_points =
            toml::find_or(s, "periphery_stresslet_max_points", stkfmm.periphery_stresslet_max_points);
    }

    if (pt.contains("fiber_periphery_interaction")) {
        const auto fp = pt.at("fiber_periphery_interaction");
        fiber_periphery_interaction.f_0 = toml::find_or(fp, "f_0", 20.0);
        fiber_periphery_interaction.lambda = toml::find_or(fp, "lambda", 0.5);
    }

    shell_precompute_file = toml::find_or(pt, "shell_precompute_file", "");
}
