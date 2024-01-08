#!/usr/bin/env python3

import numpy as np
import scipy.linalg as scla
from scipy.spatial import ConvexHull
import sys
import toml
import copy
import warnings

from skelly_sim.shape_gallery import ShapeGallery
import skelly_sim.Smooth_Closed_Surface_Quadrature_RBF as quadlib
import skelly_sim.periphery as periphery
import skelly_sim.quaternion as quaternion
import skelly_sim.kernels as kernels

def main():
    config_file = 'skelly_config.toml'
    if len(sys.argv) == 1:
        print("Using default toml file for input: 'skelly_config.toml'. "
              "Provide an alternative filename argument to this script to use that instead.")
    else:
        config_file = sys.argv[1]

    config = toml.load(config_file)

    body_quadrature_radius_offset_low = 0.1
    body_quadrature_radius_offset_high = 0.2
    body_quadrature_radius_threshold = 2.0

    # There are two effective 'radii' for periphery: the attachment radius (where fibers attach),
    # and the actual node positions. For the periphery, the node radius is scaled relative to the
    # input attachment radius (or other geometrical parameters) by this factor
    periphery_node_scale_factor = 1.04


    def precompute_periphery(config : dict):
        if 'periphery' not in config:
            return
        shell_precompute_file = config['periphery']['precompute_file']
        periphery_type = config['periphery']['shape']

        # Build shape
        if periphery_type == 'sphere':
            n_periphery = config['periphery']['n_nodes']
            np.set_printoptions(precision=16)
            periphery_radius = config['periphery']['radius'] * periphery_node_scale_factor
            print(f"CJE: Internal radius in precompute.py for sphere: {periphery_radius}")
            boundary = ShapeGallery(
                periphery_type,
                n_periphery,
                radius=periphery_radius,
            )
            print(f"CJE:boundary.nodes:     {boundary.nodes}")
            print(f"CJE:boundary.normals:   {boundary.node_normals}")
        elif periphery_type == 'ellipsoid':
            n_periphery = config['periphery']['n_nodes']
            periphery_a = config['periphery']['a'] * periphery_node_scale_factor
            periphery_b = config['periphery']['b'] * periphery_node_scale_factor
            periphery_c = config['periphery']['c'] * periphery_node_scale_factor

            boundary = ShapeGallery(
                periphery_type,
                n_periphery,
                a=periphery_a,
                b=periphery_b,
                c=periphery_c,
            )
        elif periphery_type == 'surface_of_revolution':
            envelope_config = config['periphery']['envelope']
            boundary = ShapeGallery(
                periphery_type,
                0,
                envelope_config=envelope_config,
                scale_factor=periphery_node_scale_factor,
            )
            config['periphery']['n_nodes'] = boundary.nodes.shape[0]
        elif periphery_type == 'triangulated_surface':
            triangulated_filename  = config['periphery']['triangulated_filename']
            boundary = ShapeGallery(
                periphery_type,
                0,
                triangulated_filename=triangulated_filename,
            )
            config['periphery']['n_nodes'] = boundary.nodes.shape[0]
        else:
            print("Invalid periphery: '" + periphery_type + "'")
            sys.exit()

        # Normals are in the opposite direction to bodies' normals
        nodes_periphery = boundary.nodes
        normals_periphery = -boundary.node_normals

        hull_periphery = ConvexHull(nodes_periphery)
        triangles_periphery = hull_periphery.simplices

        print(f"CJE:hull_periphery.points:      {hull_periphery.points}")
        print(f"CJE:hull_periphery.simplices:   {hull_periphery.simplices}")
        # Get quadratures
        print('Building Quadrature Weights (Periphery)')

        # This RBF function is an autogenerated mess and it spits out numeric warnings we're going
        # to ignore as to not scare the crap out of the uninformed user
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            quadrature_weights_periphery = \
                quadlib.Smooth_Closed_Surface_Quadrature_RBF(
                    nodes_periphery, triangles_periphery, boundary.h, boundary.gradh
                )
        print('Finished building Quadrature Weights (Periphery)')

        print('Creating periphery object')
        # Build shell class
        shell = periphery.Periphery(np.array([0., 0., 0.]), quaternion.Quaternion([1.0, 0.0, 0.0, 0.0]), nodes_periphery,
                                    normals_periphery, quadrature_weights_periphery)
        print('Finished creating periphery object')

        # Compute singularity subtraction vectors
        shell.get_singularity_subtraction_vectors(eta=1.0)

        # Precompute shell's r_vectors and normals
        trg_shell_surf = shell.get_r_vectors()
        normals_shell = shell.get_normals()

        # Build shell preconditioner
        N = shell.Nblobs
        weights = shell.quadrature_weights
        shell_stresslet = kernels.stresslet_kernel_times_normal_numba(trg_shell_surf, normals_shell, eta=1.0)

        I = np.zeros(shape=(3 * N, 3 * N))
        for i in range(N):
            I[3 * i:3 * (i + 1), 3 * i + 0] = shell.ex[3 * i:3 * (i + 1)] / weights[i]
            I[3 * i:3 * (i + 1), 3 * i + 1] = shell.ey[3 * i:3 * (i + 1)] / weights[i]
            I[3 * i:3 * (i + 1), 3 * i + 2] = shell.ez[3 * i:3 * (i + 1)] / weights[i]
        I_vec = np.ones(N * 3)
        I_vec[0::3] /= (1.0 * weights)
        I_vec[1::3] /= (1.0 * weights)
        I_vec[2::3] /= (1.0 * weights)
        shell_stresslet += -I - np.diag(I_vec)

        # Similarly, save shell's complementary matrix
        shell_complementary = kernels.complementary_kernel(trg_shell_surf, normals_shell)

        # Cache sum for later multiplies
        shell_stresslet_plus_complementary = shell_stresslet + shell_complementary

        # Preconditioner:
        M_inv_periphery = scla.inv(shell_stresslet_plus_complementary)

        # Singularity subtraction vectors, reshaped again
        shell.ex = shell.ex.reshape((N, 3))
        shell.ey = shell.ey.reshape((N, 3))
        shell.ez = shell.ez.reshape((N, 3))
        print("Finished periphery init.")

        with open(shell_precompute_file, 'wb') as f:
            np.savez(f,
                     quadrature_weights=quadrature_weights_periphery,
                     stresslet_plus_complementary=shell_stresslet_plus_complementary,
                     M_inv=M_inv_periphery,
                     normals=normals_periphery,
                     nodes=nodes_periphery,
                     **boundary.envelope.get_state())


    def precompute_body_sphere(body: dict):
        precompute_file = body['precompute_file']
        n_nodes = body['n_nodes']

        radius = body['radius']
        if radius < body_quadrature_radius_threshold:
            radius -= body_quadrature_radius_offset_low
        else:
            radius -= body_quadrature_radius_offset_high

        boundary = ShapeGallery(
            'sphere',
            n_nodes,
            radius=radius,
        )

        # Normals are in the opposite direction to bodies' normals
        node_positions_ref = boundary.nodes
        node_normals_ref = boundary.node_normals
        node_hull = ConvexHull(node_positions_ref)
        node_triangles = node_hull.simplices
        # Get quadratures
        print('Building Quadrature Weights (SphericalBody)')

        # This RBF function is an autogenerated mess and it spits out numeric warnings we're going
        # to ignore as to not scare the crap out of the uninformed user
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            node_weights = \
                quadlib.Smooth_Closed_Surface_Quadrature_RBF(
                    node_positions_ref, node_triangles, boundary.h, boundary.gradh
                )

        print('Finished building Quadrature Weights (SphericalBody)')

        with open(precompute_file, 'wb') as f:
            np.savez(f, node_weights=node_weights, node_normals_ref=node_normals_ref, node_positions_ref=node_positions_ref)


    def precompute_body_deformable(body: dict):
        print("Deformable body precompute not implemented")

    def precompute_body_ellipsoid(body: dict):
        precompute_file = body['precompute_file']
        n_nodes = body['n_nodes']

        # Look at each radius independently for an ellipsoid
        radius = body['axis_length']
        if radius[0] < body_quadrature_radius_threshold:
            radius[0] -= body_quadrature_radius_offset_low
        else:
            radius[0] -= body_quadrature_radius_offset_high

        if radius[1] < body_quadrature_radius_threshold:
            radius[1] -= body_quadrature_radius_offset_low
        else:
            radius[1] -= body_quadrature_radius_offset_high

        if radius[2] < body_quadrature_radius_threshold:
            radius[2] -= body_quadrature_radius_offset_low
        else:
            radius[2] -= body_quadrature_radius_offset_high

        body_a = radius[0]
        body_b = radius[1]
        body_c = radius[2]
        boundary = ShapeGallery(
            'ellipsoid',
            n_nodes,
            a=body_a,
            b=body_b,
            c=body_c,
        )

        # Normals are in the opposite direction to bodies' normals
        node_positions_ref = boundary.nodes
        node_normals_ref = boundary.node_normals
        node_hull = ConvexHull(node_positions_ref)
        node_triangles = node_hull.simplices
        # Get quadratures
        print('Building Quadrature Weights (EllipsoidalBody)')

        # This RBF function is an autogenerated mess and it spits out numeric warnings we're going
        # to ignore as to not scare the crap out of the uninformed user
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            node_weights = \
                quadlib.Smooth_Closed_Surface_Quadrature_RBF(
                    node_positions_ref, node_triangles, boundary.h, boundary.gradh
                )

        print('Finished building Quadrature Weights (EllipsoidalBody)')

        with open(precompute_file, 'wb') as f:
            np.savez(f, node_weights=node_weights, node_normals_ref=node_normals_ref, node_positions_ref=node_positions_ref)

    def precompute_body(body: dict):
        body_shape = body['shape']

        # Build shape
        if body_shape == 'sphere':
            precompute_body_sphere(body)
        elif body_shape == 'deformable':
            precompute_body_deformable(body)
        elif body_shape == 'ellipsoid':
            precompute_body_ellipsoid(body)
        else:
            print("Invalid body shape: " + body_shape)
            sys.exit()


    visited_precomputes = []
    if "bodies" in config:
        for b in config["bodies"]:
            if b['precompute_file'] not in visited_precomputes:
                visited_precomputes.append(b['precompute_file'])
                print(b)
                precompute_body(b)

    config_orig = copy.deepcopy(config)
    precompute_periphery(config)

    if (config_orig != config):
        print(
            "Config changed dynamically. This is not a problem, and is an expected output of peripheries like surface_of_revolution "
            "which dynamically figure out the number of nodes. Backing up and updating input config.")
        import shutil
        shutil.copy(config_file, config_file + '.bak')
        with open(config_file, 'w') as fh:
            toml.dump(config, fh)
