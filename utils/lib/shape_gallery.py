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

    elif shape == 'oocyte':
        # Constants and nodes
        target_nodes = Number_of_Nodes
        N_x = int(round(np.sqrt(target_nodes)))
        length = kwargs.get('length')
        T = kwargs.get('T')
        p1 = kwargs.get('p1')
        p2 = kwargs.get('p2')

        def height(x):
            return 0.5 * T * ((1 + 2*x/length)**p1) * ((1 - 2*x/length)**p2) * length

        def h(p):
            return height(p[:, 0])**2 - p[:, 1]**2 - p[:, 2]**2

        def gradh(points):
            normvec = np.zeros(shape=(len(points), 3))
            for i in range(len(points)):
                x, y, z = points[i]
                if np.abs(x + 0.5 * length) < 1E-12:
                    normvec[i, :] = np.array([-1.0, 0.0, 0.0])
                elif np.abs(x - 0.5 * length) < 1E-12:
                    normvec[i, :] = np.array([1.0, 0.0, 0.0])
                else:
                    h = np.sqrt(y**2 + z**2)
                    dh = h * 2 * (p1 / (length + 2*x) - p2 / (length - 2*x))

                    normvec[i, :] = -np.array([h * dh, -y, -z])
                    normvec[i, :] = normvec[i, :] / np.linalg.norm(normvec[i, :])
            return normvec

        # FIXME: Bonkers sampling because stupid function's slope is -infinity as x->L/2
        x = np.hstack([np.linspace(-0.5*length, 0.5*length, 100000), [0.5*length]])
        r = height(x)
        xd = np.diff(x)
        rd = np.diff(r)
        dist = np.sqrt(xd**2+rd**2)
        u = np.hstack([[0.0], np.cumsum(dist)])

        t = np.linspace(0, u.max(), N_x)
        xn = np.interp(t, u, x)
        rn = height(xn)

        ds = np.mean(np.sqrt(np.diff(xn)**2+np.diff(rn)**2))
        points = [[xn[0], 0.0, 0.0]]
        for i in range(1, len(xn)-1):
            N_radial = int(round(2 * np.pi * rn[i] / ds))
            for j in range(N_radial):
                theta = j * 2 * np.pi / N_radial
                points.append([xn[i], rn[i] * np.cos(theta), rn[i] * np.sin(theta)])

        points.append([xn[-1], 0.0, 0.0])
        quadrature_nodes = np.array(points)
        nodes_normal = gradh(points)

    return quadrature_nodes, nodes_normal, h, gradh
