import numpy as np

from pathlib import Path
from skelly_sim.reader import TrajectoryReader
from skelly_sim.skelly_config import Config, Body
from skelly_sim.testing import working_directory, run_sim, run_precompute, sim_path

np.random.seed(100)
n_nodes = 1000
config_file = 'skelly_config.toml'

# This test is based off of Kim and Karilla, pg. 64 equations for a prolate spheriod under torque
# Note that the torque is about the main symmetry axis (rotating in this case around z)

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

    # Generate an axis length for a,b,c to get a 3:1 ratio, accounting for the shifts in radii
    body_a = 1.1
    body_b = 1.1
    body_c = 3.2

    config.bodies = [
        Body(n_nucleation_sites=0,
             position=[0.0, 0.0, 0.0],
             shape='ellipsoid',
             axis_length=[body_a, body_b, body_c],
             n_nodes=n_nodes,
             external_torque=[0.0, 0.0, 0.6])
    ]

    config.save(path / config_file)
    return True

def angular_velocity(path: Path=Path('.')):
    print("Running analysis")
    with working_directory(path):
        traj = TrajectoryReader(config_file)

        # need beginning/end positions to calculate average velocity
        traj.load_frame(0)
        omega_initial = traj['bodies'][0]['solution_vec_'][-3:]
        traj.load_frame(len(traj) - 1)
        omega_final = traj['bodies'][0]['solution_vec_'][-3:]

        # Use the same radii as above, minus the correction factors
        a = 3.0
        c = 1.0
        eccentricity = np.sqrt(a**2 - c**2)/a
        print("Eccentricity:  {}".format(eccentricity))
        eta = traj.config_data['params']['eta']
        torque = np.linalg.norm(traj.config_data['bodies'][0]['external_torque'])

        # Calculate
        oneme2 = (1.0 - np.power(eccentricity,2))
        Le = np.log((1.0+eccentricity)/(1.0-eccentricity))
        Xc = (4./3.)*np.power(eccentricity,3)*(oneme2) / (2*eccentricity - oneme2*Le)

        w_theoretical = torque / (8. * np.pi * eta * a**3 * Xc)
        w_initial = omega_initial[2]

        error = abs(1 - w_initial / w_theoretical)

        print("Measured angular velocity:           {}".format(w_initial))
        print("Theoretical angular velocity         {}".format(w_theoretical))
        print("Error |1 - w/w0|:                    {}".format(error))

        return error


def test_gen_config(sim_path):
    assert gen_config(sim_path)

def test_run_precompute(sim_path):
    assert run_precompute(sim_path)

def test_run_sim(sim_path):
    assert run_sim(sim_path)

def test_angular_velocity(sim_path):
    assert angular_velocity(sim_path) < 1E-5
