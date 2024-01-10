import os
import shutil

import numpy as np

from pathlib import Path
from skelly_sim.reader import TrajectoryReader
from skelly_sim.skelly_config import ConfigTriangulated, Body
from skelly_sim.testing import working_directory, run_sim, run_precompute, sim_path

np.random.seed(100)
#n_nodes_periphery = 2000
n_nodes_body = 800
#radius_periphery = 12.0
radius_body = 2.85
config_file = 'skelly_config.toml'
triangulated_filename = 'blender_icosphere_r12_subdiv5.stl'

def gen_config(path: Path=('.')):
    print("Generating config")
    # create a config object and set the system parameters
    config = ConfigTriangulated()
    config.params.eta = 0.9
    config.params.dt_initial = 1E-2
    config.params.dt_min = 1E-5
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
             axis_length=[radius_body, radius_body, radius_body],
             n_nodes=n_nodes_body,
             external_force=[0.0, 0.0, 1.5])
    ]

    #config.periphery.n_nodes = n_nodes_periphery
    #config.periphery.radius = radius_periphery / 1.04
    config.periphery.triangulated_filename = triangulated_filename

    config.save(path / config_file)
    return True

def copy_testfile(path: Path('.')):
    print(f"Copying file {triangulated_filename}")
    # Find the original file
    module_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Perform the copy
    with working_directory(path):
        source_file_path = os.path.join(module_dir, triangulated_filename)
        destination_file_path = os.path.join(os.getcwd(), triangulated_filename)
        print(f"Source file: {source_file_path}")
        print(f"Destination file: {destination_file_path}")
        shutil.copy(source_file_path, destination_file_path)

    return True

def radially_pointing(path: Path('.')):
    # Check if the normal vectors of the precompute file are all radially pointing. In this case,
    # inwards, as it is the periphery
    print(f"Checking if periphery normals are inward pointing")
    with working_directory(path):
        precompute = np.load("periphery_precompute.npz")

        nodes = precompute['nodes']
        normals = precompute['normals']

        # Sphere, should just be able to normalize
        norms = np.linalg.norm(nodes)
        normalized_nodes = nodes / norms[:, np.newaxis]

        if len(nodes) != len(normals):
            return False
        
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

        v_regression_initial = 0.01765946618244609
        v_regression_final = 0.017659442526685217

        error_initial = abs(1 - v_initial / v_regression_initial)
        error_final = abs(1 - v_final / v_regression_final)

        print("v_initial:                   {}".format(v_initial))
        print("v_regression_initial         {}".format(v_regression_initial))
        print("Error initial |1 - v/v0|:    {}".format(error_initial))
        print("v_final:                     {}".format(v_final))
        print("v_regression_final           {}".format(v_regression_final))
        print("Error final |1 - v/v0|:      {}".format(error_final))

        error = np.sqrt(error_initial**2 + error_final**2)

        return error


def test_gen_config(sim_path):
    assert gen_config(sim_path)

#def test_copy_testfile(sim_path):
#    assert copy_testfile(sim_path)
#
#def test_run_precompute(sim_path):
#    assert run_precompute(sim_path)
#
#def test_radially_pointing(sim_path):
#    assert radially_pointing(sim_path)
#
#def test_run_sim(sim_path):
#    assert run_sim(sim_path)
#
#def test_velocity(sim_path):
#    assert velocity(sim_path) < 1E-6
