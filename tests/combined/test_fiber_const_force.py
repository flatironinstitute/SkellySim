import numpy as np

from pathlib import Path

from skelly_sim.reader import TrajectoryReader
from skelly_sim.skelly_config import Config, Fiber
from skelly_sim.testing import working_directory, sim_path, run_sim

np.random.seed(100)
n_nodes = 800
config_file = 'skelly_config.toml'

def gen_config(path: Path):
    print("Generating config")
    # create a config object and set the system parameters
    config = Config()
    config.params.eta = 0.7
    config.params.dt_initial = 1E-4
    config.params.dt_min = 1E-4
    config.params.dt_max = 1E-4
    config.params.dt_write = 1E-3
    config.params.t_final = 1E-2
    config.params.gmres_tol = 1E-10
    config.params.seed = 130319
    config.params.pair_evaluator = "CPU"

    length = 0.75
    config.fibers = [Fiber(
        force_scale=0.31,
        length=length,
        n_nodes=8,
        bending_rigidity=0.0025
    )]
    config.fibers[0].fill_node_positions(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))

    config.save(path / config_file)

    return True


def velocity(path: Path=Path('.')):
    print("Comparing velocity to analytic result")
    with working_directory(path):
        traj = TrajectoryReader('skelly_config.toml')

        traj.load_frame(0)
        x0 = traj['fibers'][0]['x_'][0, :]
        traj.load_frame(-1)
        xf = traj['fibers'][0]['x_'][0, :]

        dt = traj.times[-1] - traj.times[0]
        v = (xf - x0) / dt

        fib = traj.config_data['fibers'][0]
        length = fib['length']
        radius = fib['radius']

        epsilon = radius / length

        gamma = fib["force_scale"] * length / v[-1]
        gamma_theory = -4 * np.pi * length * traj.config_data['params']['eta'] / (np.log(np.exp(1) * epsilon**2))
        rel_error = abs(1 - gamma/gamma_theory)

        print("theoretical drag: {}".format(gamma_theory))
        print("measured drag: {}".format(gamma))
        print("relative error: {}".format(rel_error))

        return rel_error


def test_gen_config(sim_path):
    assert gen_config(sim_path)

def test_run_sim(sim_path):
    assert run_sim(sim_path)

def test_velocity(sim_path):
    assert velocity(sim_path) < 1E-6
