import numpy as np

from pathlib import Path
from skelly_sim.reader import TrajectoryReader
from skelly_sim.skelly_config import Config, Body
from skelly_sim.testing import working_directory, run_sim, run_precompute, sim_path

np.random.seed(100)
n_nodes = 800
config_file = 'skelly_config.toml'

def gen_config(path):
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
             shape='sphere',
             radius=0.5,
             n_nodes=n_nodes,
             external_force=[0.0, 0.0, 0.0],
             external_torque=[0.0, 0.0, 1.2]
             )
    ]

    config.save(path / config_file)
    return True

def angular_velocity(path: Path=Path('.')):
    print("Running analysis")
    with working_directory(path):
        traj = TrajectoryReader(config_file)

        # need beginning/end positions to calculate average velocity
        traj.load_frame(0)
        Theta_initial = traj['bodies'][0]['orientation_']
        traj.load_frame(len(traj) - 1)
        Theta_final = traj['bodies'][0]['orientation_']

        precompute_data = np.load(traj.config_data['bodies'][0]['precompute_file'])
        radius = np.linalg.norm(precompute_data["node_positions_ref"][0])
        eta = traj.config_data['params']['eta']
        torque = np.linalg.norm(traj.config_data['bodies'][0]['external_torque'])
        dt = traj.times[-1] - traj.times[0]

        w_theoretical = torque / (8 * np.pi * eta * radius**3)
        w_measured = 2 * np.arccos(np.dot(Theta_initial, Theta_final)) / dt

        return np.abs(1 - w_measured / w_theoretical)


def test_gen_config(sim_path):
    assert gen_config(sim_path)

def test_run_precompute(sim_path):
    assert run_precompute(sim_path)

def test_run_sim(sim_path):
    assert run_sim(sim_path)

def test_angular_velocity(sim_path):
    assert angular_velocity(sim_path) < 1E-6
