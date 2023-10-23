import numpy as np
from scipy.signal import find_peaks

from pathlib import Path

from skelly_sim.reader import TrajectoryReader, Listener, Request
from skelly_sim.skelly_config import ConfigSpherical, Body, Fiber
from skelly_sim.testing import working_directory, sim_path, run_precompute, run_sim

np.random.seed(100)
config_file = 'skelly_config.toml'

def gen_config(path: Path):
    print("Generating config")
    # create a config object and set the system parameters
    config = ConfigSpherical()
    config.params.eta = 1.0
    config.params.dt_initial = 5E-2
    config.params.dt_min = 1E-4
    config.params.dt_max = 5E-2
    config.params.dt_write = 5E-2
    config.params.t_final = 5.0
    config.params.gmres_tol = 1E-8
    config.params.seed = 130319
    config.params.pair_evaluator = "FMM"
    config.params.adaptive_timestep_flag = True

    body_radius = 0.5
    body_origin = np.array([0.0, 0.0, 2.2])
    nucleation_site = np.array([0.0, 0.0, body_radius])

    config.bodies = [
        Body(
            n_nucleation_sites=1,
            nucleation_sites=nucleation_site.tolist(),
            position=body_origin.tolist(),
            shape='sphere',
            radius=body_radius,
            n_nodes=400,
            external_force_type='linear',
            external_force=[0.0, 0.0, 1.0],
            )
        ]

    flength = 1.0
    n_nodes = 32
    config.fibers = [
        Fiber(
            parent_body=0,
            parent_site=0,
            n_nodes=n_nodes,
            )
        ]

    config.fibers[0].x = np.linspace(body_origin + nucleation_site,
                                     body_origin + nucleation_site + np.array([0.0, 0.0, flength]),
                                     n_nodes).ravel().tolist()

    config.periphery.n_nodes = 2000
    config.periphery.radius = 4.0 / 1.04

    config.params.periphery_binding.active = True
    config.params.periphery_binding.threshold = 0.1

    config.save(path / config_file)

    return True

def final_positions(path: Path=Path('.')):
    filepath = Path(__file__).parent.resolve()
    last_map = np.load(filepath / "fdfiber_compression_finalpositions.npz")
    with working_directory(path):
        traj = TrajectoryReader('skelly_config.toml')
        x_all = []
        y_all = []
        z_all = []
        body_pos = []
        for i in range(len(traj)):
            traj.load_frame(i)
            r_fib = traj['fibers'][0]['x_']
            x_all.append(r_fib[:, 0])
            y_all.append(r_fib[:, 1])
            z_all.append(r_fib[:, 2])
            body_pos.append(traj['bodies'][0]['position_'])

        # Alias the last known positions from the file
        xlast_known = last_map['xlast']
        ylast_known = last_map['ylast']
        zlast_known = last_map['zlast']
        bodylast_known = last_map['bodylast']

        # Test against known positions
        x_rel_error = np.abs(1.0 - x_all[-1]/xlast_known)
        x_tot_error = np.sqrt(np.square(x_rel_error).sum())
        y_rel_error = np.abs(1.0 - y_all[-1]/ylast_known)
        y_tot_error = np.sqrt(np.square(y_rel_error).sum())
        z_rel_error = np.abs(1.0 - z_all[-1]/zlast_known)
        z_tot_error = np.sqrt(np.square(z_rel_error).sum())
        b_rel_error = np.abs(1.0 - body_pos[-1]/bodylast_known)
        b_tot_error = np.sqrt(np.square(b_rel_error).sum())

        tot_error = np.sqrt(np.square(x_tot_error) + np.square(y_tot_error) + np.square(z_tot_error) + np.square(b_tot_error))

        print("Body and fiber position total error: {}".format(tot_error))

    return tot_error

def velocity_field_err(path: Path=Path('.')):
    filepath = Path(__file__).parent.resolve()
    velocity_field = np.load(filepath / "fdfiber_compression_data.npy")
    velocity_field_flat = velocity_field.flatten()

    with working_directory(path):
        traj = TrajectoryReader('skelly_config.toml')
        shell_radius = traj.config_data['periphery']['radius']
        body_radius = traj.config_data['bodies'][0]['radius']
        body_pos = np.empty(shape=(len(traj), 3))
        for i in range(len(traj)):
            traj.load_frame(i)
            body_pos[i,:] = traj['bodies'][0]['position_']

        # Fire up SkellySim in listener mode
        listener = Listener(binary='skelly_sim')

        # Requests are done via a Request object
        req = Request()

        # Specify what we want
        req.frame_no = 98
        req_evaluator = "FMM"

        # Request the velocity field
        tmp = np.linspace(-shell_radius, shell_radius, 20)
        xm, ym, zm = np.meshgrid(tmp, tmp, tmp)
        xcube = np.array((xm.ravel(), ym.ravel(), zm.ravel())).T

        # Filter out points outside the periphery and inside the body
        relpoints = np.where((np.linalg.norm(xcube - body_pos[98,:], axis=1) > body_radius) & (np.linalg.norm(xcube, axis=1) < shell_radius))
        req.velocity_field.x = xcube[relpoints]

        # Make our listener request
        res = listener.request(req)

        x = req.velocity_field.x
        v = res['velocity_field']

        v_flat = v.flatten()

        # Do a comparison amongst all terms
        rel_error = np.abs(1.0 - v_flat/velocity_field_flat)
        tot_error = np.sqrt(np.square(rel_error).sum())

        print("Velocity field total error: {}".format(tot_error))

    return tot_error

def test_gen_config(sim_path):
    assert gen_config(sim_path)

def test_run_precompute(sim_path):
    assert run_precompute(sim_path)

def test_run_sim(sim_path):
    assert run_sim(sim_path)

def test_last_position(sim_path):
    assert final_positions(sim_path) < 1E-5

def test_last_velocity_field(sim_path):
    assert velocity_field_err(sim_path) < 1E-5

