import numpy as np

from pathlib import Path
from skelly_sim.reader import TrajectoryReader
from skelly_sim.skelly_config import Config, Body
from skelly_sim.testing import working_directory, run_sim, run_precompute, sim_path

np.random.seed(100)
n_nodes = 800
config_file = 'skelly_config.toml'

def gen_config(path: Path=('.')):
    print("Generating config")
    # create a config object and set the system parameters
    config = Config()
    config.params.eta = 0.9
    config.params.dt_initial = 1E-1
    config.params.dt_min = 1E-4
    config.params.dt_max = 1E-1
    config.params.dt_write = 1E-1
    config.params.t_final = 1.5
    config.params.gmres_tol = 1E-10
    config.params.seed = 130319
    config.params.pair_evaluator = "CPU"

    # Generate an axis length for a,b that is slightly smaller than before
    smalleps = 0.1
    radius = 0.5
    body_a = radius * (1.0 - smalleps)
    body_b = radius * (1.0 - smalleps)
    body_c = radius

    config.bodies = [
        Body(n_nucleation_sites=0,
             position=[0.0, 0.0, 0.0],
             shape='ellipsoid',
             axis_length=[body_a, body_b, body_c],
             n_nodes=n_nodes,
             external_force=[0.0, 0.0, 1.5])
    ]

    config.save(path / config_file)
    return True

def velocity(path: Path=Path('.')):
    print("Running analysis")
    with working_directory(path):
        traj = TrajectoryReader(config_file)

        # need beginning/end positions to calculate average velocity
        traj.load_frame(0)
        z_initial = traj['bodies'][0]['position_'][2]
        traj.load_frame(len(traj) - 1)
        z_final = traj['bodies'][0]['position_'][2]

        dt = traj.times[-1] - traj.times[0]

        # We need the hydrodynamic radius, which is slightly different than the supplied 'attachment'
        # radius. let's just grab it from the precompute data.
        precompute_data = np.load(traj.config_data['bodies'][0]['precompute_file'])
        # Compute all the radii for the ellipsoid, then extract the two axes
        radii = np.zeros(len(precompute_data['node_positions_ref']))
        for idx in range(len(radii)):
            radii[idx] = np.linalg.norm(precompute_data['node_positions_ref'][idx,:])

        a = np.max(radii)
        b = np.min(radii)
        # Derive back smalleps due to changes in the hydrodynamic radius that occur under the hood
        print("Effective radius a:      {}".format(a))
        print("Effective radius b:      {}".format(b))
        smalleps = 1.0 - b/a
        eccentricity = np.sqrt(1.0 - (b/a)**2)
        print("Effective epsilon:       {}".format(smalleps))
        print("Effective eccentricity:  {}".format(eccentricity))
        eta = traj.config_data['params']['eta']
        force = traj.config_data['bodies'][0]['external_force'][2]

        v_parallel = force / (16.0 * np.pi * eta * a * np.power(eccentricity,3) / ((1 + np.power(eccentricity,2))*np.log((1+eccentricity)/(1-eccentricity))-2*eccentricity))
        v_theoretical_sphere_large = force / (6.0 * np.pi * eta * a)
        v_theoretical_sphere_small = force / (6.0 * np.pi * eta * b)
        v_measured = (z_final - z_initial) / dt
        error = abs(1 - v_measured / v_parallel)

        print("Measured velocity:                   {}".format(v_measured))
        print("Theoretical velocity (parallel):     {}".format(v_parallel))
        print("Theoretical velocity (small sphere): {}".format(v_theoretical_sphere_small))
        print("Theoretical velocity (large sphere): {}".format(v_theoretical_sphere_large))
        print("Error |1 - v/v0|:                    {}".format(error))

        return error


def test_gen_config(sim_path):
    assert gen_config(sim_path)

def test_run_precompute(sim_path):
    assert run_precompute(sim_path)

def test_run_sim(sim_path):
    assert run_sim(sim_path)

def test_velocity(sim_path):
    assert velocity(sim_path) < 1E-3
