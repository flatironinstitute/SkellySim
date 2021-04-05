#include <params.hpp>

#include <mpi.h>

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
    seed = toml::find_or(pt, "seed", 1);

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

    shell_precompute_file = toml::find_or(pt, "shell_precompute_file", "");
}
