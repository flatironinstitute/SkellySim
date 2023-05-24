import numpy as np
from scipy.signal import find_peaks

from pathlib import Path

from skelly_sim.reader import TrajectoryReader
from skelly_sim.skelly_config import Config, Fiber, Point
from skelly_sim.testing import working_directory, sim_path, run_sim

np.random.seed(100)
config_file = 'skelly_config.toml'

def gen_config(path: Path):
    print("Generating config")
    # create a config object and set the system parameters
    config = Config()
    config.params.eta = 1.0
    config.params.dt_initial = 0.02
    config.params.dt_min = 0.01
    config.params.dt_max = 0.1
    config.params.dt_write = 0.1
    config.params.t_final = 50.0
    config.params.gmres_tol = 1E-10
    config.params.seed = 130319
    config.params.pair_evaluator = "CPU"
    config.params.adaptive_timestep_flag = True

    sigma = 80.0
    length = 1.0
    bending_rigidity = 0.0025
    force_scale = -sigma * bending_rigidity / length**3
    n_nodes = 32

    config.fibers = [Fiber(
        force_scale=force_scale,
        length=length,
        n_nodes=n_nodes,
        bending_rigidity=bending_rigidity,
        minus_clamped=True
    )]

    # Give a small starting kick
    config.point_sources = [
        Point(
            position=[0.0, 0.0, 10*length],
            force=[10.0, 0.0, 0.0],
            time_to_live=1.0,
        )
    ]

    # Orient in Z
    config.fibers[0].x = np.linspace([0, 0, 0], [0, 0, length], n_nodes).ravel().tolist()

    config.save(path / config_file)

    return True

def is_decaying(path: Path=Path('.')):
    with working_directory(path):
        traj = TrajectoryReader('skelly_config.toml')

        x = []
        for i in range(len(traj)):
            traj.load_frame(i)
            x.append(traj['fibers'][0]['x_'][-1, 0])

        x = np.array(x)
        mpeaks, _ = find_peaks(x, height=0)

        # Check the peaks, ignoring the first one from the original 'kick' to the system
        x_peak1 = x[mpeaks[1]]
        x_peak2 = x[mpeaks[2]]
        print(f"peak 1 x: {x_peak1}")
        print(f"peak 2 x: {x_peak2}")

        # Just return if the solution is decaying at this timescale
        return x_peak2 < x_peak1

def compare_previous_peaks(path: Path=Path('.')):
    print(f"Comparing deflection to previous results")
    with working_directory(path):
        traj = TrajectoryReader('skelly_config.toml')

        x = []
        for i in range(len(traj)):
            traj.load_frame(i)
            x.append(traj['fibers'][0]['x_'][-1, 0])

        x = np.array(x)
        mpeaks, _ = find_peaks(x, height=0)

        # Check the peaks, ignoring the first one from the original 'kick' to the system, compared to previous results
        x_peak1 = x[mpeaks[1]]
        x_peak2 = x[mpeaks[2]]

        x_peak1_test = 0.09575812
        x_peak2_test = 0.13564472

        rel_error_1 = abs(1 - x_peak1/x_peak1_test)
        rel_error_2 = abs(1 - x_peak2/x_peak2_test)
        rel_error = np.sqrt(np.power(rel_error_1, 2) + np.power(rel_error_2, 2))

        print(f"peak 1 x: {x_peak1}")
        print(f"peak 2 x: {x_peak2}")
        print(f"relative error peak 1: {rel_error_1}")
        print(f"relative error peak 2: {rel_error_2}")
        print(f"relative error: {rel_error}")

        return rel_error

def test_gen_config(sim_path):
    assert gen_config(sim_path)

def test_run_sim(sim_path):
    assert run_sim(sim_path)

def test_is_decaying(sim_path):
    assert not is_decaying(sim_path)

def test_compare_previous_peaks(sim_path):
    assert compare_previous_peaks(sim_path) < 1E-6

