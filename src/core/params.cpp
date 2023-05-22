#include <params.hpp>

Params::Params(toml::value &pt) {
    eta = toml::find_or(pt, "eta", 1.0);
    dt_initial = toml::find_or(pt, "dt_initial", 1E-2);
    dt_min = toml::find_or(pt, "dt_min", 1E-4);
    dt_max = toml::find_or(pt, "dt_max", 2.0);
    beta_up = toml::find_or(pt, "beta_up", 1.2);
    beta_down = toml::find_or(pt, "beta_down", 0.5);
    adaptive_timestep_flag = toml::find_or(pt, "adaptive_timestep_flag", true);
    dt_write = toml::find_or(pt, "dt_write", 0.25);
    t_final = toml::find_or(pt, "t_final", 1.0);
    gmres_tol = toml::find_or(pt, "gmres_tol", 1E-10);
    fiber_error_tol = toml::find_or(pt, "fiber_error_tol", 1E-1);
    seed = toml::find_or(pt, "seed", 1);
    implicit_motor_activation_delay = toml::find_or(pt, "implicit_motor_activation_delay", 0.0);
    periphery_interaction_flag = toml::find_or(pt, "periphery_interaction_flag", false);
    pair_evaluator = toml::find_or(pt, "pair_evaluator", "FMM");

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
            dynamic_instability.radius = toml::find(di, "radius").as_floating();
            dynamic_instability.min_length = toml::find(di, "min_length").as_floating();
            dynamic_instability.bending_rigidity = toml::find(di, "bending_rigidity").as_floating();
            dynamic_instability.min_separation = toml::find(di, "min_separation").as_floating();
        } catch (std::runtime_error &e) {
            MPI_Finalize();
            exit(1);
        }
    }

    if (pt.contains("periphery_binding")) {
        const auto &pb = pt.at("periphery_binding");
        try {
            periphery_binding.active = toml::find(pb, "active").as_boolean();
            periphery_binding.polar_angle_start = toml::find(pb, "polar_angle_start").as_floating();
            periphery_binding.polar_angle_end = toml::find(pb, "polar_angle_end").as_floating();
            periphery_binding.threshold = toml::find(pb, "threshold").as_floating();

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
        fiber_periphery_interaction.f_0 = toml::find_or(fp, "f_0", fiber_periphery_interaction.f_0);
        fiber_periphery_interaction.l_0 = toml::find_or(fp, "l_0", fiber_periphery_interaction.l_0);
    }
}

void Params::print() {
    // Print out the information that we have on the system (only one, don't do it to global)
    // XXX: Whenever you add a new variable, make sure to also add a print statement here!
    spdlog::info("****** SkellySim {} ({}) ******", SKELLYSIM_VERSION, SKELLYSIM_COMMIT);
    spdlog::info("******    Parameters     ******");
    spdlog::info("eta                               = {}", eta);
    spdlog::info("dt_initial                        = {}", dt_initial);
    spdlog::info("dt_min                            = {}", dt_min);
    spdlog::info("dt_max                            = {}", dt_max);
    spdlog::info("beta_up                           = {}", beta_up);
    spdlog::info("beta_down                         = {}", beta_down);
    spdlog::info("adaptive_timestep_flag            = {}", adaptive_timestep_flag);
    spdlog::info("dt_write                          = {}", dt_write);
    spdlog::info("t_final                           = {}", t_final);
    spdlog::info("gmres_tol                         = {}", gmres_tol);
    spdlog::info("fiber_error_tol                   = {}", fiber_error_tol);
    spdlog::info("seed                              = {}", seed);
    spdlog::info("implicit_motor_activation_delay   = {}", implicit_motor_activation_delay);
    spdlog::info("periphery_interaction_flag        = {}", periphery_interaction_flag);
    spdlog::info("pair_evaluator                    = {}", pair_evaluator);
}
