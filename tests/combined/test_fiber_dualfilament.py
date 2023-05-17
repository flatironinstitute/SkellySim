import numpy as np

from pathlib import Path

from skelly_sim.reader import TrajectoryReader
from skelly_sim.skelly_config import Config, Fiber, perturbed_fiber_positions
from skelly_sim.testing import working_directory, sim_path, run_sim

np.random.seed(100)
config_file = 'skelly_config.toml'

def gen_config(path: Path):
    print("Generating config")
    # create a config object and set the system parameters
    config = Config()
    config.params.eta = 1.0
    config.params.dt_initial = 1E-1
    config.params.dt_write = 1.0
    config.params.t_final = 1E1
    config.params.gmres_tol = 1E-10
    config.params.seed = 130319
    config.params.pair_evaluator = "CPU"
    config.params.adaptive_timestep_flag = False

    sigma = 0.0225
    length = 2.0
    bending_rigidity = 0.0025
    n_nodes = 64

    n_fibers = 2

    config.fibers = [Fiber(
        force_scale=-sigma,
        length=length,
        n_nodes=n_nodes,
        bending_rigidity=bending_rigidity,
        minus_clamped=True
    ) for i in range(n_fibers)]

    # Disrupt the first fiber only
    x = perturbed_fiber_positions(0.01, length, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), n_nodes, np.array([1.0, 0.0, 0.0]))
    config.fibers[0].x = x.ravel().tolist()
    config.fibers[1].fill_node_positions(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))

    config.save(path / config_file)

    return True


def deflection(path: Path=Path('.')):
    print(f"Comparing deflection to previous results")
    with working_directory(path):
        traj = TrajectoryReader('skelly_config.toml')

        traj.load_frame(-1)
        x0 = traj['fibers'][0]['x_'][-1,0]
        x1 = traj['fibers'][1]['x_'][-1,0]

        # The deflection of the fiber(s) from previous results with these parameter choices
        x0_test = -0.004765810967995735
        x1_test = 1.0048647877439878

        rel_error_0 = abs(1 - x0/x0_test)
        rel_error_1 = abs(1 - x1/x1_test)
        rel_error = np.sqrt(np.power(rel_error_0, 2) + np.power(rel_error_1, 2))

        print(f"x0: {x0}")
        print(f"x1: {x1}")
        print(f"relative error x0 (driver):   {rel_error_0}")
        print(f"relative error x1 (response): {rel_error_1}")
        print(f"relative error: {rel_error}")

        return rel_error


def test_gen_config(sim_path):
    assert gen_config(sim_path)

def test_run_sim(sim_path):
    assert run_sim(sim_path)

def test_deflection(sim_path):
    assert deflection(sim_path) < 1E-6
