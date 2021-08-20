import numpy as np
import scipy.linalg as scla
from scipy.spatial import ConvexHull
import sys
import toml

import lib.shape_gallery as shape_gallery
import lib.Smooth_Closed_Surface_Quadrature_RBF as quadlib
import lib.periphery as periphery
import lib.quaternion as quaternion
import lib.kernels as kernels

if len(sys.argv) != 2:
    print("No input file supplied. Please supply input 'toml' file.")
    sys.exit()

config = toml.load(sys.argv[1])

body_quadrature_radius_offset_low = 0.1
body_quadrature_radius_offset_high = 0.2
body_quadrature_radius_threshold = 2.0


# There are two effective 'radii' for periphery: the attachment radius (where fibers attach),
# and the actual node positions. For the periphery, the node radius is scaled relative to the
# input attachment radius (or other geometrical parameters) by this factor
periphery_node_scale_factor = 1.04

def precompute_periphery(config):
    if 'periphery' not in config:
        return
    shell_precompute_file = config['params']['shell_precompute_file']
    periphery_type = config['periphery']['shape']
    n_periphery = config['periphery']['n_nodes']
    eta = config['params']['eta']

    # Build shape
    if periphery_type == 'sphere':
        periphery_radius = config['periphery']['radius'] * periphery_node_scale_factor
        nodes_periphery, normals_periphery, h_periphery, gradh_periphery = \
            shape_gallery.shape_gallery(
                periphery_type,
                n_periphery,
                radius=periphery_radius,
            )
    elif periphery_type == 'ellipsoid':
        periphery_a = config['periphery']['a'] * periphery_node_scale_factor
        periphery_b = config['periphery']['b'] * periphery_node_scale_factor
        periphery_c = config['periphery']['c'] * periphery_node_scale_factor

        nodes_periphery, normals_periphery, h_periphery, gradh_periphery = \
            shape_gallery.shape_gallery(
                periphery_type,
                n_periphery,
                a=periphery_a,
                b=periphery_b,
                c=periphery_c,
            )
    elif periphery_type == 'oocyte':
        T = config['periphery']['T']
        p1 = config['periphery']['p1']
        p2 = config['periphery']['p2']
        length = config['periphery']['length']
        nodes_periphery, normals_periphery, h_periphery, gradh_periphery = \
            shape_gallery.shape_gallery(
                periphery_type,
                n_periphery,
                T=T,
                p1=p1,
                p2=p2,
                length=length,
            )
    else:
        print("Invalid periphery " + periphery_type)
        sys.exit()

    # Normals are in the opposite direction to bodies' normals
    normals_periphery = -normals_periphery
    hull_periphery = ConvexHull(nodes_periphery)
    triangles_periphery = hull_periphery.simplices
    # Get quadratures
    print('Building Quadrature Weights')
    quadrature_weights_periphery = \
        quadlib.Smooth_Closed_Surface_Quadrature_RBF(
            nodes_periphery, triangles_periphery, h_periphery, gradh_periphery
        )
    print('Finished building Quadrature Weights')

    print('Creating periphery object')
    # Build shell class
    shell = periphery.Periphery(np.array([0., 0., 0.]), quaternion.Quaternion([1.0, 0.0, 0.0, 0.0]),
                                nodes_periphery, normals_periphery, quadrature_weights_periphery)
    print('Finished creating periphery object')

    # Compute singularity subtraction vectors
    shell.get_singularity_subtraction_vectors(eta=eta)

    # Precompute shell's r_vectors and normals
    trg_shell_surf = shell.get_r_vectors()
    normals_shell = shell.get_normals()

    # Build shell preconditioner
    N = shell.Nblobs
    weights = shell.quadrature_weights
    shell_stresslet = kernels.stresslet_kernel_times_normal_numba(trg_shell_surf, normals_shell, eta=eta)

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
    print(M_inv_periphery)

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
                 nodes=nodes_periphery)

def precompute_body_sphere(body):
    precompute_file = body['precompute_file']
    num_nodes = body['num_nodes']

    radius = body['radius']
    if radius < body_quadrature_radius_threshold:
        radius -= body_quadrature_radius_offset_low
    else:
        radius -= body_quadrature_radius_offset_high

    node_positions_ref, node_normals_ref, h_body, gradh_body = \
        shape_gallery.shape_gallery(
            'sphere',
            num_nodes,
            radius=radius,
        )

    # Normals are in the opposite direction to bodies' normals
    node_hull = ConvexHull(node_positions_ref)
    node_triangles = node_hull.simplices
    # Get quadratures
    print('Building Quadrature Weights')
    node_weights = \
        quadlib.Smooth_Closed_Surface_Quadrature_RBF(
            node_positions_ref, node_triangles, h_body, gradh_body
        )
    print('Finished building Quadrature Weights')

    with open(precompute_file, 'wb') as f:
        np.savez(f,
                 node_weights=node_weights,
                 node_normals_ref=node_normals_ref,
                 node_positions_ref=node_positions_ref)

def precompute_body_deformable(body: dict):
    print("Deformable body precompute not implemented")

def precompute_body(body: dict):
    body_shape = body['shape']

    # Build shape
    if body_shape == 'sphere':
        precompute_body_sphere(body)
    elif body_shape == 'deformable':
        precompute_body_deformable(body)
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

precompute_periphery(config)
