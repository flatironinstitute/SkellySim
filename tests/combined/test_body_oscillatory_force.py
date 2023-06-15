import numpy as np

from pathlib import Path
from skelly_sim.reader import TrajectoryReader
from skelly_sim.skelly_config import ConfigSpherical, Body
from skelly_sim.testing import working_directory, run_sim, run_precompute, sim_path

np.random.seed(100)
config_file = 'skelly_config.toml'

def gen_config(path: Path=('.')):
    print("Generating config")
    # create a config object and set the system parameters
    config = ConfigSpherical()
    config.params.dt_initial = 5E-2
    config.params.dt_min = 1E-4
    config.params.dt_max = 5E-2
    config.params.dt_write = 5E-2
    config.params.t_final = 10.0
    config.params.gmres_tol = 1E-10
    config.params.seed = 130319
    config.params.pair_evaluator = "CPU"

    config.bodies = [
        Body(n_nucleation_sites=0,
             position=[0.0, 0.0, 0.0],
             shape='sphere',
             radius=0.5,
             n_nodes=400,
             external_force_type='Oscillatory',
             external_force=[0.0, 0.0, 1.0],
             external_oscillation_force_amplitude=2.0,
             external_oscillation_force_frequency=0.1,
             external_oscillation_force_phase=0.0)
    ]

    config.periphery.n_nodes = 500
    config.periphery.radius = 4.0/1.04

    config.save(path / config_file)
    return True

def z_final_position(path: Path=Path('.')):
    print("Running analysis")
    with working_directory(path):
        traj = TrajectoryReader(config_file)

        # Get start/end positions
        traj.load_frame(0)
        z_initial = traj['bodies'][0]['position_'][2]
        traj.load_frame(len(traj) - 1)
        z_final = traj['bodies'][0]['position_'][2]

        # Since this is an oscillatory force, check where the sphere returned to
        z_final_theoretical = 6.215481243240294E-05
        error = abs(1 - z_final / z_final_theoretical)
        print(f"Measured final position:    {z_final}")
        print(f"Theoretical final position: {z_final_theoretical}")
        print(f"Error:                      {z_final}")

        return error


def test_gen_config(sim_path):
    assert gen_config(sim_path)

def test_run_precompute(sim_path):
    assert run_precompute(sim_path)

def test_run_sim(sim_path):
    assert run_sim(sim_path)

def test_velocity(sim_path):
    assert z_final_position(sim_path) < 1E-6
