from __future__ import division, print_function
import numpy as np
from functools import partial
from function_generator import FunctionGenerator


class Envelope(FunctionGenerator):
    def __init__(self, config):
        self.lower_bound_target = config['lower_bound']
        self.upper_bound_target = config['upper_bound']

        import numpy as np
        locals().update(config)

        self.raw_height_func = eval("lambda x: " + config['height'], locals())

        delta = 1E-10
        lb = self.lower_bound_target
        ub = self.upper_bound_target
        while delta < 0.005 * (ub - lb):
            try:
                super().__init__(self.raw_height_func, lb, ub)
            except Exception:
                lb = self.lower_bound_target + delta
                ub = self.upper_bound_target - delta
                delta *= 10
                continue
            break

        try:
            self.a
        except NameError:
            raise RuntimeError("Unable to fit height function")
        print("Height fit succeeded with bounds", (lb, ub))

    def export(self, filename):
        with open(filename, 'wb') as fh:
            np.savez(fh, n=self.n, a=self.a, b=self.b,
                     lbs=self.lbs, ubs=self.ubs, coef_mat=self.coef_mat,
                     bounds_table=self.bounds_table)


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

    elif shape == 'surface_of_revolution':
        envelope_config = kwargs.get('envelope_config')
        envelope = Envelope(envelope_config)

        # Constants and nodes
        target_nodes = envelope_config['n_nodes_target']
        N_x = int(round(np.sqrt(target_nodes)))

        x = np.hstack([np.linspace(envelope.lower_bound_target, envelope.upper_bound_target, 1000000), [envelope.upper_bound_target]])
        r = envelope.raw_height_func(x)
        xd = np.diff(x)
        rd = np.diff(r)
        dist = np.sqrt(xd**2+rd**2)
        u = np.hstack([[0.0], np.cumsum(dist)])

        t = np.linspace(0, u.max(), N_x)
        xn = np.interp(t, u, x)
        rn = envelope.raw_height_func(xn)

        ds = np.mean(np.sqrt(np.diff(xn)**2+np.diff(rn)**2))
        points = []
        for i in range(0, len(xn)):
            N_radial = int(round(2 * np.pi * rn[i] / ds))
            if N_radial <= 1:
                points.append([xn[i], 0.0, 0.0])
                continue
            for j in range(N_radial):
                theta = j * 2 * np.pi / N_radial
                points.append([xn[i], rn[i] * np.cos(theta), rn[i] * np.sin(theta)])

        def h(points):
            return envelope.raw_height_func(points[:, 0])**2 - points[:, 1]**2 - points[:, 2]**2

        def gradh(points):
            normvec = np.zeros(shape=(len(points), 3))
            for i in range(len(points)):
                x, y, z = points[i]
                if x < envelope.a:
                    normvec[i, :] = np.array([-1.0, 0.0, 0.0])
                elif x > envelope.b:
                    normvec[i, :] = np.array([1.0, 0.0, 0.0])
                else:
                    h = envelope(x)
                    dh = envelope.differentiate(x)

                    normvec[i, :] = -np.array([h * dh, -y, -z])
                    normvec[i, :] = normvec[i, :] / np.linalg.norm(normvec[i, :])
            return normvec

        quadrature_nodes = np.array(points)
        nodes_normal = gradh(points)

    return quadrature_nodes, nodes_normal, h, gradh
