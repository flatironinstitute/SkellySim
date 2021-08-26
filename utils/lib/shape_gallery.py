import numpy as np
from functools import partial
from function_generator import FunctionGenerator


class Envelope(FunctionGenerator):
    def __init__(self, config=None):
        if not config:
            self.init = False
            return
        self.init = True
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

    def get_state(self):
        if not self.init:
            return {}
        else:
            return {
                'fg_n': self.n,
                'fg_a': self.a,
                'fg_b': self.b,
                'fg_lbs': self.lbs,
                'fg_ubs': self.ubs,
                'fg_coef_mat': self.coef_mat,
                'fg_bounds_table': self.bounds_table,
            }


class ShapeGallery:
    def __init__(self, shape, n_nodes, *args, **kwargs):
        self.nodes: np.ndarray = None
        self.node_normals: np.ndarray = None
        self.h = None
        self.gradh = None
        self.envelope: Envelope = Envelope()

        if shape == 'sphere':
            # Constants and nodes
            phi = (1 + np.sqrt(5)) / 2
            N = n_nodes // 2
            radius = kwargs.get('radius')
            nodes = np.zeros((n_nodes, 3))
            node_normals = np.zeros((n_nodes, 3))

            # Fill nodes
            for i in range(-N, N):
                lat = np.arcsin((2.0 * i) / (2 * N + 1))
                lon = (i % phi) * 2 * np.pi / phi
                if lon < -np.pi:
                    lon = 2 * np.pi + lon
                elif lon > np.pi:
                    lon = lon - 2 * np.pi
                nodes[i + N, :] = [np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)]
            nodes *= radius

            # Define level surface and gradient
            def h(p):
                return p[:, 0] * p[:, 0] \
                    + p[:, 1] * p[:, 1] \
                    + p[:, 2] * p[:, 2] \
                    - radius * radius

            def gradh(p):
                return 2 * p

            # Compute surface normal at nodes
            node_normals = gradh(nodes)
            node_normals /= np.linalg.norm(node_normals, axis=1, keepdims=True)

            self.nodes = nodes
            self.node_normals = node_normals
            self.h = h
            self.gradh = gradh

        elif shape == 'ellipsoid':
            # Constants and nodes
            phi = (1 + np.sqrt(5)) / 2
            N = n_nodes // 2
            a = kwargs.get('a')
            b = kwargs.get('b')
            c = kwargs.get('c')
            nodes = np.zeros((n_nodes, 3))
            node_normals = np.zeros((n_nodes, 3))

            # Fill nodes
            for i in range(-N, N):
                lat = np.arcsin((2.0 * i) / (2 * N + 1))
                lon = (i % phi) * 2 * np.pi / phi
                if lon < -np.pi:
                    lon = 2 * np.pi + lon
                elif lon > np.pi:
                    lon = lon - 2 * np.pi
                nodes[i + N, :] = [a * np.cos(lon) * np.cos(lat), b * np.sin(lon) * np.cos(lat), c * np.sin(lat)]

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
            node_normals = gradh(nodes)
            node_normals /= np.linalg.norm(node_normals, axis=1, keepdims=True)

            self.nodes = nodes
            self.node_normals = node_normals
            self.h = h
            self.gradh = gradh

        elif shape == 'surface_of_revolution':
            envelope_config = kwargs.get('envelope_config')
            envelope = Envelope(envelope_config)

            # Number of 'rings' to represent envelope should be roughly the square root of the target
            target_nodes = envelope_config['n_nodes_target']
            N_x = int(round(np.sqrt(target_nodes)))

            # Evaluate function finely so that we can get a good estimate for the arc length
            x = np.hstack([np.linspace(envelope.lower_bound_target, envelope.upper_bound_target, 1000000),
                           [envelope.upper_bound_target]])
            r = envelope.raw_height_func(x)
            xd = np.diff(x)
            rd = np.diff(r)
            # arc length segment lengths
            dist = np.sqrt(xd**2+rd**2)
            # total arc length along as function of x
            u = np.hstack([[0.0], np.cumsum(dist)])

            # Now we sample evenly along the length of the curve, so that points 't' are equispaced in 's'
            # Then just solve for the 'x' values that give those 't' values
            t = np.linspace(0, u.max(), N_x)
            xn = np.interp(t, u, x)
            rn = envelope.raw_height_func(xn)

            # Draw rings around each point x along the x axis, with of radius h(x)
            # Attempt to have roughly the same 'ds' along the ring as along 'x'
            ds = np.mean(np.sqrt(np.diff(xn)**2+np.diff(rn)**2))
            nodes = []
            for i in range(0, len(xn)):
                N_radial = int(round(2 * np.pi * rn[i] / ds))
                if N_radial <= 1:
                    nodes.append([xn[i], 0.0, 0.0])
                    continue
                for j in range(N_radial):
                    theta = j * 2 * np.pi / N_radial
                    nodes.append([xn[i], rn[i] * np.cos(theta), rn[i] * np.sin(theta)])

            def h(nodes):
                return envelope.raw_height_func(nodes[:, 0])**2 - nodes[:, 1]**2 - nodes[:, 2]**2

            def gradh(nodes):
                normvec = np.zeros(shape=(len(nodes), 3))
                for i in range(len(nodes)):
                    x, y, z = nodes[i]
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

            nodes = np.array(nodes)
            node_normals = gradh(nodes)

            self.nodes = nodes
            self.node_normals = node_normals
            self.h = h
            self.gradh = gradh
            self.envelope = envelope
