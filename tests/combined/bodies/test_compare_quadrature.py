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
    config.params.t_final = 0.2
    config.params.gmres_tol = 1E-10
    config.params.seed = 130319
    config.params.pair_evaluator = "CPU"

    config.bodies = [
        Body(n_nucleation_sites=0,
             position=[0.0, 0.0, 0.0],
             shape='ellipsoid',
             axis_length=[0.5, 0.5, 0.5],
             n_nodes=n_nodes,
             external_force=[0.0, 0.0, 1.5],
             precompute_file='ellipsoidal_precompute.npz'),
        Body(n_nucleation_sites=0,
             position=[5.0, 0.0, 0.0],
             shape='sphere',
             radius=0.5,
             n_nodes=n_nodes,
             external_force=[0.0, 0.0, 1.5],
             precompute_file='sphere_precompute.npz'),
    ]

    config.save(path / config_file)
    return True

def compare_quadrature(path: Path=Path('.')):
    print("Comparing quadrature between a sphere and a spherical ellipsoid (also sphere)")
    with working_directory(path):
        # Get the precompute data to compare the quadrature, etc
        precompute_ellipsoid    = np.load("ellipsoidal_precompute.npz")
        precompute_sphere       = np.load("sphere_precompute.npz")

        # Compare node weights
        weights_compare = np.allclose(precompute_ellipsoid['node_weights'], precompute_sphere['node_weights'], atol=1e-12)

        # Compare node normals
        normals_compare = np.allclose(precompute_ellipsoid['node_normals_ref'], precompute_sphere['node_normals_ref'], atol=1e-12)

        # Compare node positions
        positions_compare = np.allclose(precompute_ellipsoid['node_positions_ref'], precompute_sphere['node_positions_ref'], atol=1e-12)

        return (weights_compare and normals_compare and positions_compare)

def test_gen_config(sim_path):
    assert gen_config(sim_path)

def test_run_precompute(sim_path):
    assert run_precompute(sim_path)

def test_quadrature(sim_path):
    assert compare_quadrature(sim_path)

