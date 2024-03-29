import numpy as np

from pathlib import Path
from skelly_sim.reader import TrajectoryReader
from skelly_sim.skelly_config import Config, Body
from skelly_sim.testing import working_directory, run_sim, run_precompute, sim_path

np.random.seed(100)
#n_nodes = 2000
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
    config.params.t_final = 1.0
    config.params.gmres_tol = 1E-10
    config.params.seed = 130319
    config.params.pair_evaluator = "CPU"

    config.bodies = [
        Body(n_nucleation_sites=0,
             position=[0.0, 0.0, 0.0],
             shape='ellipsoid',
             axis_length=[0.5, 0.5, 0.5],
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
        v_initial = traj['bodies'][0]['solution_vec_'][-4]
        traj.load_frame(len(traj) - 1)
        z_final = traj['bodies'][0]['position_'][2]
        v_final = traj['bodies'][0]['solution_vec_'][-4]

        dt = traj.times[-1] - traj.times[0]

        # We need the hydrodynamic radius, which is slightly different than the supplied 'attachment'
        # radius. let's just grab it from the precompute data.
        precompute_data = np.load(traj.config_data['bodies'][0]['precompute_file'])

        # Find the radius from all of the node_positions_ref, and look for the max (should be around 0.4)
        radii = np.zeros(len(precompute_data['node_positions_ref']))
        for idx in range(len(radii)):
            radii[idx] = np.linalg.norm(precompute_data['node_positions_ref'][idx,:])
        radius = np.mean(radii)
        print("Effective radius XY: {}".format(radius))
        eta = traj.config_data['params']['eta']
        force = traj.config_data['bodies'][0]['external_force'][2]

        v_theoretical = force / (6 * np.pi * eta * radius)
        v_measured = (z_final - z_initial) / dt
        error = abs(1 - v_measured / v_theoretical)

        print("Measured velocity:    {}".format(v_measured))
        print("v_initial:            {}".format(v_initial))
        print("v_final:              {}".format(v_final))
        print("Theoretical velocity: {}".format(v_theoretical))
        print("Error |1 - v/v0|:     {}".format(error))

        return error


def test_gen_config(sim_path):
    assert gen_config(sim_path)

def test_run_precompute(sim_path):
    assert run_precompute(sim_path)

def test_run_sim(sim_path):
    assert run_sim(sim_path)

def test_velocity(sim_path):
    assert velocity(sim_path) < 1E-6
