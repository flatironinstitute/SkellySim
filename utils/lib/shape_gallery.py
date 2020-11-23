from __future__ import division, print_function
import numpy as np
from functools import partial


def shape_gallery(shape, Number_of_Nodes, *args, **kwargs):
    if shape == 'sphere':
        # Constants and nodes
        phi = (1 + np.sqrt(5)) / 2
        N = Number_of_Nodes // 2
        radius = kwargs.get('radius')
        quadrature_nodes = np.zeros((Number_of_Nodes, 3))
        nodes_normal = np.zeros((Number_of_Nodes, 3))

        # Fill nodes
        for i in range(-N, N):
            lat = np.arcsin((2.0 * i) / (2 * N + 1))
            lon = (i % phi) * 2 * np.pi / phi
            if lon < -np.pi:
                lon = 2 * np.pi + lon
            elif lon > np.pi:
                lon = lon - 2 * np.pi
            quadrature_nodes[i + N, :] = [np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)]
        quadrature_nodes *= radius

        # Define level surface and gradient
        def h(p):
            return p[:, 0] * p[:, 0] \
                + p[:, 1] * p[:, 1] \
                + p[:, 2] * p[:, 2] \
                - radius * radius

        def gradh(p):
            return 2 * p

        # Compute surface normal at nodes
        nodes_normal = gradh(quadrature_nodes)
        nodes_normal /= np.linalg.norm(nodes_normal, axis=1, keepdims=True)

    elif shape == 'ellipsoid':
        # Constants and nodes
        phi = (1 + np.sqrt(5)) / 2
        N = Number_of_Nodes // 2
        a = kwargs.get('a')
        b = kwargs.get('b')
        c = kwargs.get('c')
        quadrature_nodes = np.zeros((Number_of_Nodes, 3))
        nodes_normal = np.zeros((Number_of_Nodes, 3))

        # Fill nodes
        for i in range(-N, N):
            lat = np.arcsin((2.0 * i) / (2 * N + 1))
            lon = (i % phi) * 2 * np.pi / phi
            if lon < -np.pi:
                lon = 2 * np.pi + lon
            elif lon > np.pi:
                lon = lon - 2 * np.pi
            quadrature_nodes[i + N, :] = [a * np.cos(lon) * np.cos(lat), b * np.sin(lon) * np.cos(lat), c * np.sin(lat)]

        # Define level surface
        def h_func(p, abc):
            return (p[:, 0] / abc[0])**2 + (p[:, 1] / abc[1])**2 + (p[:, 2] / abc[2])**2 - 1.0

        h = partial(h_func, abc=np.array([a, b, c]))

        # Define gradient
        def gradh_func(p, abc):
            """
            Gradient for an ellipsoid x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
            Pass a, b, c constants as abc = np.array([a, b, c]).
            """
            return 2 * p / abc**2

        gradh = partial(gradh_func, abc=np.array([a, b, c]))

        # Compute surface normal at nodes
        nodes_normal = gradh(quadrature_nodes)
        nodes_normal /= np.linalg.norm(nodes_normal, axis=1, keepdims=True)

    return quadrature_nodes, nodes_normal, h, gradh
