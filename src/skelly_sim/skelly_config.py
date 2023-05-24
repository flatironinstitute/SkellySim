from typing import List, Tuple, Callable
from dataclasses import dataclass, asdict, field, is_dataclass
from argparse import Namespace
import numpy as np
from scipy.special import ellipeinc, ellipe
from scipy.optimize import fsolve, bisect
import toml
from skelly_sim import shape_gallery, param_tools
from dataclass_utils import check_type


def _ellipsoid(t: float, u: float, a: float, b: float, c: float):
    """
    Return point on ellipsoid(t=angle1, u=angle2) parameterized by the axes [a, b, c].

    Returns
    -------
    np.array(3)
        Point on the surface of the ellipsoid given input parameters
    """
    return np.array([a * np.sin(u) * np.cos(t), b * np.sin(u) * np.sin(t), c * np.cos(u)])


def _build_cdf(f: Callable[[float], float], lb: float, ub: float) -> Tuple[np.array, np.array]:
    """
    Builds a discretized cumulative distribution function of the arclength along the curve defined by f(x) on [lb, ub].
    Useful for distributing random positions uniformly along a curve (i.e. for placing fibers on an arbitrary surface).
    Used with sister function invert_cdf

    Returns
    -------
    tuple(np.array, np.array)
        uniformly separated x_i and its associated cdf(x_i)
    """

    # Evaluate function finely so that we can get a good estimate for the arc length
    xs = np.hstack([np.linspace(lb, ub, 1000000), [ub]])
    rs = f(xs)
    xd = np.diff(xs)
    rd = np.diff(rs)
    # shell area (off by some constant factor. using mean of height since diff() shortens
    # vectors)
    dist = np.sqrt(xd**2 + rd**2) * (rs[0:-1] + rs[1:])
    # cumulative arc length as function of x
    u = np.hstack([[0.0], np.cumsum(dist)]) / np.sum(dist)
    return xs, u


def _invert_cdf(y: float, xs: np.array, u: np.array) -> float:
    """
    Solves the cdf equation 'cdf(x) = y' for x using bisection. Meant to be used with the output from _build_cdf

    Returns
    -------
    float
        'x' value s.t. 'cdf(x) ~= y'
    """

    def f(x):
        return y - np.interp(x, xs, u)

    return bisect(f, xs[0], xs[-1])


def _get_random_point_on_sphere():
    """
    Give a uniform random point on the surface of a unit sphere

    Returns: numpy 3d-vector of unit length on surface of a unit sphere
    """
    phi = np.random.uniform() * 2.0 * np.pi
    u = 2 * np.random.uniform() - 1.0
    factor = np.sqrt(1.0 - u * u)

    return np.array([np.cos(phi) * factor, np.sin(phi) * factor, u])


def _sin_length(amplitude: float, xf: float):
    """
    Compute the arc length of the function amplitude * sin(2*pi*x/xf) on [0, xf]

    Arguments
    ---------
    amplitude : float
        amplitude of sine wave
    xf : float
        upper bound of arc length integral
    """
    A2 = (2 * np.pi * amplitude / xf)**2
    return xf / np.pi * (ellipe(-A2) + np.sqrt(1 + A2) * ellipe(A2 / (1 + A2)))


def _cos_length_full(amplitude: float, xi: float, xf: float, x_max: float):
    """
    The arclength of amplitude * cos(2*pi*x/x_max) on the interval [xi, xf]
    Designed to be used with an fsolve on (_sin_length() - length) to get x_max
    Arclengths are bizarrely complicated!

    Arguments
    ---------
    amplitude : float
        amplitude of sine wave
    xi : float
        lower bound of integral
    xf : float
        upper bound of integral
    x_max : float
        maximum xf for our given length (basically where one period is reached)
    """
    scale_factor = 2.0 * np.pi / x_max
    A2 = (scale_factor * amplitude)**2
    return (ellipeinc(scale_factor * xf, -A2) - ellipeinc(scale_factor * xi, -A2)) / scale_factor


def _get_random_orthogonal_vector(x: np.array):
    """Take an input 3d vector 'x' and return a random unit vector orthogonal to the input vector"""

    if x[1] != 0 or x[2] != 0:
        offaxis = np.array([1, 0, 0])
    else:
        offaxis = np.array([0, 1, 0])
    b = np.cross(x, offaxis)
    b /= np.linalg.norm(b)
    c = np.cross(x, b)

    theta = 2 * np.pi * np.random.uniform()
    return b * np.cos(theta) + c * np.sin(theta)


def perturbed_fiber_positions(amplitude: float, length: float, x0: np.array, normal: np.array, n_nodes: int, ortho: np.array = None):
    """
    Create a fiber x vector with a small cosine perturbation in a random direction orthogonal to the fiber.

    Arguments
    ---------
    amplitude : float
        Amplitude of the perturbation. Make it a small fraction of the total length
    length : float
        Length of the fiber
    x0 : np.array(3)
        3D position of the base of the Fiber. Don't place at the origin since the orientation is determined from this.
    normal : np.array(3)
        Direction fiber points at the two tips of the perturbation
    n_nodes : int
        number of nodes to represent the fiber
    ortho : np.array(3)
        Normalized 3D orientation vector for the perturbation. Not sanity checked (that it's actually normal, or length 1)

    Returns: [n_nodes, 3] matrix of positions of each node in a perturbed fiber
    """
    x_max_fun = lambda xf: _sin_length(amplitude, xf) - length
    x_max = fsolve(x_max_fun, length)

    if ortho is None:
        ortho = _get_random_orthogonal_vector(normal)

    arc_length_per_segment = length / (n_nodes - 1)
    lin_pos = np.zeros(n_nodes)
    for i in range(1, n_nodes):
        fun = lambda xf: _cos_length_full(amplitude, lin_pos[i - 1], xf, x_max) - arc_length_per_segment
        lin_pos[i] = fsolve(fun, lin_pos[i - 1] + arc_length_per_segment)

    fiber_positions = np.outer(lin_pos, normal)

    cos_perturbation = amplitude * (np.cos(2 * np.pi * lin_pos / lin_pos[-1]) - 1)
    fiber_positions += np.outer(cos_perturbation, ortho)
    fiber_positions += x0

    return fiber_positions


def _unpack(obj):
    """
    Helper routine to process a Config object recursively into the format needed by skelly_sim
    Used in toml dumping.

    Arguments
    ---------
    At entry, it should probably be a dataclass Config object

    Returns
    -------
    Dict
        Dictionary representation of our nested dataclass for toml export
    """
    if isinstance(obj, dict):
        return {key: _unpack(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        if not obj:
            return
        return [_unpack(value) for value in obj]
    elif is_dataclass(obj):
        return {key: _unpack(value) for key, value in asdict(obj).items()}
    elif isinstance(obj, Namespace):
        return _unpack(dict(filter(lambda keyval: not keyval[0].startswith('__'), obj.__dict__.items())))
    elif isinstance(obj, tuple):
        return tuple(_unpack(value) for value in obj)
    else:
        return obj


def _check_invalid_attributes(obj, errorstatus=False):
    """
    Helper routine to process a Config object recursively looking for invalid attributes.

    Arguments
    ---------
    At entry, it should probably be a dataclass Config object

    Returns
    -------
    bool
        True if there is an invalid attribute, False otherwise
    """
    if is_dataclass(obj):
        instance_keys = set(obj.__dict__.keys())
        class_keys = set(obj.__dataclass_fields__.keys())
        extra_keys = instance_keys.difference(class_keys)
        if extra_keys:
            for key in extra_keys:
                print("Error: Attribute '{}' not found in class '{}'".format(key, type(obj).__name__))
            errorstatus = True
        for _, value in obj.__dict__.items():
            errorstatus = _check_invalid_attributes(value, errorstatus)
    elif isinstance(obj, list):
        for value in obj:
            errorstatus = _check_invalid_attributes(value, errorstatus)
    return errorstatus


def _default_vector():
    """
    A default vector factory for dataclass 'field' objects: :obj:`[0.0, 0.0, 0.0]`
    """
    return [0.0, 0.0, 0.0]


def _default_ivector():
    """
    A default integer vector factory for dataclass 'field' objects: :obj:`[0, 0, 0]`
    """
    return [0, 0, 0]


def _default_quaternion():
    """
    A default quaternion factory for dataclass 'field' objects: :obj:`[0.0, 0.0, 0.0, 1.0]`

    """
    return [0.0, 0.0, 0.0, 1.0]


@dataclass
class Fiber():
    """
    dataclass representing a single fiber

    Attributes
    ----------
    n_nodes : int, default: :obj:`32`
        Number of nodes to represent fiber. Highly deformed or very long fibers will require more nodes to be accurately represented
    parent_body : int, default: :obj:`-1`
        Index of :obj:`Body` the :obj:`Fiber` is bound to. A value of :obj:`-1` indicates a free fiber.
        :obj:`Fibers` attached to :obj:`Bodies` obey the :obj:`clamped` boundary condition,
        which preserves the relative angle of attachment to the body.
    force_scale : float, default: :obj:`0.0`, units: :obj:`pN·μm⁻¹`
        Tangential force per unit length to act along filament. A positive value pushes toward the :obj:`plus` end,
        while a negative value pushes toward the :obj:`minus` end
    bending_rigidity : float default: :obj:`2.5E-3`, units: :obj:`pN·μm²`
        Bending rigidity of this filament
    radius : float, default: :obj:`0.0125`, units: :obj:`μm`
        Radius of fiber (for purposes of SBT, currently no physical dimension)
    length : float, default: :obj:`1.0`, units: :obj:`μm`
        Constraint length of this filament
    minus_clamped : bool, default: :obj:`False`
        Fix minus end of filament with "clamped" boundary condition, preserving orientation and position (:obj:`Velocity = 0, AngularVelocity = 0`).
        If attached to a body (:obj:`parent_body >= 0`), then this parameter is implied True and ignored.
    x : List[float], default: :obj:`[]`, units: :obj:`μm`
        List of node positions in [x0,y0,z0,x1,y1,z1...] order. Extreme care must be taken when setting this since the length constraint
        can generate massive tensions with poor input. See examples.
    """
    n_nodes: int = 32
    parent_body: int = -1
    parent_site: int = -1
    force_scale: float = 0.0
    bending_rigidity: float = 2.5E-3
    radius: float = 0.0125
    length: float = 1.0
    minus_clamped: bool = False
    x: List[float] = field(default_factory=list)

    def fill_node_positions(self, x0: np.array, normal: np.array):
        """
        Update fiber node positions based on a linear array from x0 along normal. I.e. the fiber
        nodes are uniformly spaced on the line segment :obj:`x0 + s * normal` on :obj:`s = [0, self.length]`

        Modifies: self.x (node positions)

        Arguments
        ---------
        x0 : np.array
            3d position of fiber minus end
        normal : np.array
            normalized vector of fiber orientation
        """
        fiber_positions = x0 + \
            self.length * np.linspace(0, normal, self.n_nodes)
        self.x = fiber_positions.ravel().tolist()


@dataclass
class DynamicInstability():
    """
    dataclass for dynamic instability parameters

    Attributes
    ----------
    n_nodes : int, default: :obj:`0`
        Number of nodes for newly nucleated fibers (see nucleation_rate). :obj:`0` disables dynamic instability
    v_growth : float, default: :obj:`0.0`, units: :obj:`μm·s⁻¹`
        Growth velocity in microns/second. Growth happens at the "plus" ends
    f_catastrophe : float, default: :obj:`0.0`, units: :obj:`s⁻¹`
        Catastrophe frequency (probability per unit time of switching from growth to deletion) in 1/second
    v_grow_collision_scale : float, default: :obj:`0.5`
        When a fiber hits a boundary, scale its velocity (v_growth) by this factor
    f_catastrophe_collision_scale : float, default: :obj:`2.0`
        When a fiber hits a boundary, scale its catastrophe frequency (f_catastrophe) by this
    nucleation_rate : float, default: :obj:`0.0`, units: :obj:`s⁻¹`
        Fiber nucleation rate in units of MT nucleations / second
    radius: float, default: :obj:`0.025`, units: :obj:`μm`
        New fiber radius in microns
    min_length: float, default: :obj:`0.5`, units: :obj:`μm`
        New fiber initial length in microns
    bending_rigidity : float, default: :obj:`2.5E-3`, units: :obj:`pN·μm²`
        New fiber bending rigidity
    min_separation : float, default: :obj:`0.1`, units: :obj:`μm`
        Minimum distance between Fiber minus ends in microns when nucleating (closer than this will be rejected)
    """
    n_nodes: int = 0
    v_growth: float = 0.0
    f_catastrophe: float = 0.0
    v_grow_collision_scale: float = 0.5
    f_catastrophe_collision_scale: float = 2.0
    nucleation_rate: float = 0.0
    radius: float = 0.025
    min_length: float = 0.5
    bending_rigidity: float = 2.5E-3
    min_separation: float = 0.1


@dataclass
class PeripheryBinding():
    """
    dataclass for periphery binding parameters

    Attributes
    ----------
    active : bool, default: :obj:`False`
        Have plus ends of fibers hinge at periphery if set
    polar_angle_start : float, default: :obj:`0.0`, units: :obj:`radians`
        Minimum angle where cortex binding happens
    polar_angle_end : float, default: :obj:`pi/2`, units: :obj:`radians`
        Maximum angle where cortex binding happens
    threshold : float, default: :obj:`0.75`, units: :obj:`μm`
        Minimum distance between Fiber plus ends and cortex in microns when binding happens
    """
    active: bool = False
    polar_angle_start: float = 0.0
    polar_angle_end: float = 0.5 * np.pi
    threshold: float = 0.75


@dataclass
class Params():
    """dataclass representing system/meta parameters for the entire simulation

    Attributes
    ----------
    eta : float, default: :obj:`1.0`, units: :obj:`Pa·s`
        Viscosity of fluid
    dt_initial : float, default: :obj:`0.025`, units: :obj:`s`
        Initial length of timestep
    dt_min : float, default: :obj:`1E-5`, units: :obj:`s`
        Minimum timestep before simulation fails when using adaptive timestepping (adaptive_timestep_flag)
    dt_max : float, default: :obj:`0.1`, units: :obj:`s`
        Maximum timestep size allowable when using adaptive timestepping (adaptive_timestep_flag)
    dt_write : float, default: :obj:`0.1`, units: :obj:`s`
        Amount of simulation time between writes. Due to adaptive timestepping (and floating point issues) the
        time between writes is only approximately dt_write
    t_final : float, default: :obj:`100.0`, units: :obj:`s`
        Simulation time to quit the simulation
    gmres_tol : float, default: :obj:`1E-8`
        GMRES tolerance, might be tuned later, but recommend at least 1E-8
    fiber_error_tol : float, default: :obj:`0.1`
        Fiber error tolerance. Not recommended to tamper with.
        Fiber error is the maximum deviation between 1.0 and a the derivative along the fiber.
        When using adaptive timestepping, if the error exceeds this value, the timestep is rejected.
    periphery_binding_flag : bool, default: :obj:`False`
        If set, fiber plus ends near the periphery (closer than 0.75, hardcoded) will use
        hinged boundary conditions. Intended for use with dynamic instability
    seed : int, default: :obj:`130319`
        Random number seed at simulation runtime (doesn't affect numpy seed during configuration generation)
    dynamic_instability : DynamicInstability, default: :obj:`DynamicInstability()`
        Dynamic instability parameters
    periphery_interaction_flag : bool, default: :obj:`False`
        Experimental repulsion between periphery and Fibers
    adaptive_timestep_flag : bool, default: :obj:`True`
        If set, use adaptive timestepping, which attempts to control simulation error by reducing the timestep
        when the solution has convergence issues
    pair_evaluator : str, default: :obj:`"FMM"`
        Type of evaluator to use for kernels (stokeslet, stokes double layer, etc)
        Valid values: "CPU", "FMM"
    """
    eta: float = 1.0
    dt_initial: float = 0.025
    dt_min: float = 1E-5
    dt_max: float = 0.025
    dt_write: float = 0.1
    t_final: float = 100.0
    gmres_tol: float = 1E-8
    fiber_error_tol: float = 1E-1
    periphery_binding_flag: bool = False
    seed: int = 130319
    implicit_motor_activation_delay: float = 0.0
    dynamic_instability: DynamicInstability = field(default_factory=DynamicInstability)
    periphery_binding: PeripheryBinding = field(default_factory=PeripheryBinding)
    periphery_interaction_flag: bool = False
    adaptive_timestep_flag: bool = True
    pair_evaluator: str = "FMM"


@dataclass
class Periphery():
    """
    dataclass representing a periphery. Don't use directly, but use one of the subclasses

    Attributes
    ----------
    n_nodes : int, default: :obj:`6000`
        Number of nodes to represent Periphery object. larger peripheries = more nodes. Memory scales as n_nodes^2, so don't exceed ~10000
    precompute_file : str, default: :obj:`periphery_precompute.npz`
        File to store the periphery precompute data
    """
    n_nodes: int = 6000
    precompute_file: str = 'periphery_precompute.npz'

    def find_binding_site(self, fibers: List[Fiber]):
        """
        stub for finding a binding site on a periphery surface given a list of fibers
        """
        return None


@dataclass
class SphericalPeriphery(Periphery):
    """
    dataclass representing a spherical periphery

    Attributes
    ----------
    n_nodes : int, default: :obj:`6000`
        Number of nodes to represent Periphery object. larger peripheries = more nodes. Memory scales as n_nodes^2, so don't exceed ~10000
    shape : str, default: :obj:`'sphere'`
        Shape of the periphery. Don't modify it!
    radius : float, default: :obj:`6.0`, units: :obj:`μm`
        Radius of our sphere in microns
    """

    shape: str = 'sphere'
    radius: float = 6.0

    def find_binding_site(self, fibers: List[Fiber], ds_min) -> Tuple[np.array, np.array]:
        """
        Find an open binding site given a list of Fibers that could interfere with binding
        Binding site is assumed uniform on the surface, and placed a small epsilon away from the surface (0.9999999 * radius) to prevent
        interacting with the periphery directly. The binding site is guaranteed to be further than the Euclidean distance ds_min from any
        other Fiber minus end

        Arguments
        ---------
        fibers : list[Fiber]
            Fibers that could potentially block a binding site
        ds_min : float
            Minimum allowable separation between a binding site and any fiber minus end

        Returns
        -------
        tuple(np.array, np.array)
            position vector and its normalized version
        """
        while (True):
            u0 = _get_random_point_on_sphere()
            x0 = 0.99999999 * u0 * self.radius

            accept = True
            ds_min2 = ds_min * ds_min
            for fib in fibers:
                dx = x0 - fib.x[0:3]
                if np.dot(dx, dx) < ds_min2:
                    accept = False
                    break
            if accept:
                return (x0, u0)

    def move_fibers_to_surface(self, fibers: List[Fiber], ds_min: float, verbose: bool = True) -> None:
        """
        Take a list of fibers and randomly and uniformly place them normal to the surface with a minimum separation ds_min.

        Arguments
        ---------
        fibers : List[Fiber]
            List of fibers that will be moved. Only the Fiber.x property will be modified
        ds_min : float
            Minimum separation allowable between the fiber minus ends. Collisions are not searched for the rest of the fibers,
            though they are unlikely
        verbose : bool, default: :obj:`True`
            If true, print a progress message
        """
        print("Inserting fibers")
        for i in range(len(fibers)):
            (x0, u0) = self.find_binding_site(fibers[0:i], ds_min)
            if verbose:
                print("Inserting fiber {} at {}".format(i, x0))
            fibers[i].fill_node_positions(x0, -u0)


@dataclass
class EllipsoidalPeriphery(Periphery):
    """
    dataclass representing an ellipsoidal periphery. (x/a)^2 + (y/b)^2 + (z/c)^2 = 1

    Attributes
    ----------
    n_nodes : int, default: :obj:`6000`
        Number of nodes to represent Periphery object. larger peripheries = more nodes. Memory scales as n_nodes^2, so don't exceed ~10000
    shape : str, default: :obj:`'ellipsoid'`
        Shape of the periphery. Don't modify it!
    a : float, default: :obj:`7.8`, units: :obj:`μm`
         Length of axis 'a'
    b : float, default: :obj:`4.16`, units: :obj:`μm`
         Length of axis 'b'
    c : float, default: :obj:`4.16`, units: :obj:`μm`
         Length of axis 'c'
    """

    shape: str = 'ellipsoid'
    a: float = 7.8
    b: float = 4.16
    c: float = 4.16

    def move_fibers_to_surface(self, fibers: List[Fiber], ds_min: float, verbose: bool = True) -> None:
        """
        Take a list of fibers and randomly and uniformly place them normal to the surface with a minimum separation ds_min.

        Arguments
        ---------
        fibers : List[Fiber]
            List of fibers that will be moved. Only the Fiber.x property will be modified
        ds_min : float
            Minimum separation allowable between the fiber minus ends. Collisions are not searched for the rest of the fibers,
            though they are unlikely
        verbose : bool, default: :obj:`True`
            If true, print a progress message
        """
        print("Generating trial uniform points on surface")
        n_trials = 5 * len(fibers)

        def ellipsoid(t, u):
            return _ellipsoid(t, u, self.a / 1.04, self.b / 1.04, self.c / 1.04)

        x_trial = param_tools.r_surface(n_trials, ellipsoid, *(0, 2 * np.pi), *(0, np.pi))[0]

        print("Inserting fibers")
        ds_min2 = ds_min * ds_min
        i_trial = 0
        for i in range(len(fibers)):
            if i_trial >= n_trials:
                print("Unable to insert fibers. Add more fiber trials, or decrease fiber density on the surface.")
                sys.exit()

            fib: Fiber = fibers[i]

            i_trial_start = i_trial
            while (True):
                x0 = x_trial[:, i_trial]

                reject = False
                for j in range(0, i - 1):
                    dx = x0 - fibers[j].x[0:3]
                    if np.dot(x0, x0) < ds_min2:
                        i_trial_fiber += 1
                        reject = True
                        break
                if reject:
                    continue

                # Use our envelope function to calculate the gradient/normal
                normal = np.array([x0[0] / self.a**2, x0[1] / self.b**2, x0[2] / self.c**2])
                normal = -normal / np.linalg.norm(normal)

                fibers[i].fill_node_positions(x0, normal)

                i_trial += 1
                print("Inserted fiber {} after {} trials".format(i, i_trial - i_trial_start))
                break


@dataclass
class RevolutionPeriphery(Periphery):
    """dataclass representing a surface of revolution. The user provides an envelope function 'h(x)' (see skelly_sim.shape_gallery.Envelope)
    which is rotated around the 'x' axis to generate a periphery. Note that the 'skelly_precompute' utility will actually update
    your output toml file with the correct number of nodes. This complicated machinery is so that later on 'skelly_sim' can directly look for
    collisions using the input height function

    .. highlight:: python
    .. code-block:: python

        # Example usage:
        # Target number of nodes -- actual number likely larger
        config.periphery.envelope.n_nodes_target = 6000

        # lower/upper bound are required options. ideally your function should go to zero at the upper/lower bounds
        config.periphery.envelope.lower_bound = -3.75
        config.periphery.envelope.upper_bound = 3.75

        # required option. this is the function you're revolving around the 'x' axis. 'x' needs to be
        # the independent variable. Currently the function has to be a one-liner
        # Unit lengths should be in μm
        config.periphery.envelope.height = "0.5 * T * ((1 + 2*x/length)**p1) * ((1 - 2*x/length)**p2) * length"
        config.periphery.envelope.T = 0.72
        config.periphery.envelope.p1 = 0.4
        config.periphery.envelope.p2 = 0.2
        config.periphery.envelope.length = 7.5


    Attributes
    ----------
    n_nodes : int, default: :obj:`0`
        Number of nodes to represent Periphery object. This will be set later by the envelope
        during the precompute stage, so don't bother changing from default
    shape : str, default: :obj:`surface_of_revolution`
        Shape of the periphery. Don't modify it!
    envelope : Namespace, default: :obj:`argparse.Namespace()`
        See example above
    """

    shape: str = 'surface_of_revolution'
    n_nodes: int = 0
    envelope: Namespace = field(default_factory=Namespace)

    def move_fibers_to_surface(self, fibers: List[Fiber], ds_min: float, verbose: bool = True) -> None:
        """
        Take a list of fibers and randomly and uniformly place them normal to the surface with a minimum separation ds_min.

        Arguments
        ---------
        fibers : List[Fiber]
            List of fibers that will be moved. Only the Fiber.x property will be modified
        ds_min : float
            Minimum separation allowable between the fiber minus ends. Collisions are not searched for the rest of the fibers,
            though they are unlikely
        verbose : bool, default: :obj:`True`
            If true, print a progress message
        """
        print("Building envelope object and CDF...")
        envelope = shape_gallery.Envelope(self.envelope)

        xs, u = _build_cdf(envelope.raw_height_func, self.envelope.lower_bound, self.envelope.upper_bound)
        print("Finished building envelope object and CDF...")

        ds_min2 = ds_min * ds_min
        for i in range(len(fibers)):
            i_trial = 0
            reject = True
            while (reject):
                i_trial += 1

                # generate trial 'x'
                x_trial = _invert_cdf(np.random.uniform(), xs, u)
                h_trial = envelope.raw_height_func(x_trial)

                # generate trial 'y, z'
                theta = 2 * np.pi * np.random.uniform()
                y_trial = h_trial * np.cos(theta)
                z_trial = h_trial * np.sin(theta)

                # base of Fiber
                x0 = np.array([x_trial, y_trial, z_trial])

                # check for collisions
                reject = False
                for j in range(0, i - 1):
                    dx = x0 - fibers[j].x[0:3]
                    if np.dot(dx, dx) < ds_min2:
                        reject = True
                        break
                if reject:
                    continue

                # we need a normal, which requires derivatives. Our envelope can calculate them
                # arbitrarily.  However, the envelope object is a fit, and it doesn't necessarily fit
                # down to the supplied upper/lower bounds, if that function has a divergent
                # derivative. Here we just assume that unfit points points are aligned along 'x'
                if x0[0] < envelope.a:
                    normal = np.array([1.0, 0.0, 0.0])
                elif x0[0] > envelope.b:
                    normal = np.array([-1.0, 0.0, 0.0])
                else:
                    # Use our envelope function to calculate the gradient/normal
                    normal = np.array([envelope(x0[0]) * envelope.differentiate(x0[0]), -x0[1], -x0[2]])
                    normal = normal / np.linalg.norm(normal)

                fibers[i].fill_node_positions(x0, normal)
                if verbose:
                    print("Inserted fiber {} after {} trials".format(i, i_trial))


@dataclass
class Body():
    """dataclass for a single body and its parameters

    Attributes
    ----------
    n_nucleation_sites : int, default: :obj:`0`
        Number of available Fiber sites on the body. Don't add more fibers than this to body
    position : List[float], default: :obj:`[0.0, 0.0, 0.0]`, units: :obj:`μm`
        Lab frame coordinate of the body center [x,y,z]
    orientation : List[float], default: :obj:`[0.0, 0.0, 0.0, 1.0]`
        Orientation quaternion of the body. Not worth changing
    shape : str, default: :obj:`'sphere'`
        Shape of the body. Sphere is currently only supported option
    radius : float, default: :obj:`1.0`, units: :obj:`μm`
        Radius of the body. This is the attachment radius for nucleation sites, the hydrodynamic radius is a bit smaller
    n_nodes : int, default: :obj:`600`
        Number of nodes to represent surface. WARNING: MAKE NEW PRECOMPUTE DATA WHEN CHANGING or you will regret it.
    precompute_file : str, default: :obj:`'body_precompute.npz'`
        Where precompute data is stored (quadrature data, mostly). Can be different on
        different bodies, though should be the same if the bodies are the same radius and have
        the same numbers of nodes.
    external_force : List[float], default: :obj:`[0.0, 0.0, 0.0]`, units: :obj:`pN`
        Lab frame external force applied to body - useful for testing things like stokes flow
    """
    n_nucleation_sites: int = 0
    position: List[float] = field(default_factory=_default_vector)
    orientation: List[float] = field(default_factory=_default_quaternion)
    shape: str = 'sphere'
    radius: float = 1.0
    n_nodes: int = 600
    precompute_file: str = 'body_precompute.npz'
    external_force: List[float] = field(default_factory=_default_vector)
    external_torque: List[float] = field(default_factory=_default_vector)
    nucleation_sites: List[float] = field(default_factory=list)

    def find_binding_site(self, fibers: List[Fiber], ds_min: float) -> Tuple[np.array, np.array]:
        """
        Find an open binding site given a list of Fibers that could interfere with binding
        Binding site is assumed uniform on the surface, and placed a small epsilon away from the surface (0.9999999 * radius) to prevent
        interacting with the periphery directly. The binding site is guaranteed to be further than the Euclidean distance ds_min from any
        other Fiber minus end

        Arguments
        ---------

        ds_min : float
            Minimum allowable separation between a binding site and any fiber minus end

        Returns
        -------
        tuple(np.array, np.array)
            position vector and its normalized version
        """
        com = np.array(self.position)
        while (True):
            u0 = _get_random_point_on_sphere()
            x0 = u0 * self.radius + com

            accept = True
            ds_min2 = ds_min * ds_min
            for fib in fibers:
                dx = x0 - fib.x[0:3]
                if np.dot(dx, dx) < ds_min2:
                    accept = False
                    break
            if accept:
                return (x0, u0)


    def generate_nucleation_sites(self, ds_min: float, verbose: bool = True):
        """
        Find an open binding site given a list of Fibers that could interfere with binding
        Binding site is assumed uniform on the surface, and placed a small epsilon away from the surface (0.9999999 * radius) to prevent
        interacting with the periphery directly. The binding site is guaranteed to be further than the Euclidean distance ds_min from any
        other Fiber minus end

        Arguments
        ---------
        ds_min : float
            Minimum allowable separation between a binding site and any fiber minus end

        Returns
        -------
        tuple(np.array, np.array)
            position vector and its normalized version
        """
        com = np.array(self.position)
        ds_min2 = ds_min * ds_min

        sites = np.empty((self.n_nucleation_sites, 3))
        for isite in range(self.n_nucleation_sites):
            while (True):
                x0 = _get_random_point_on_sphere() * self.radius + com
                accept = True
                for jsite in range(isite):
                    dx = x0 - sites[isite, :]
                    if np.dot(dx, dx) < ds_min2:
                        accept = False
                        break
                if accept:
                    sites[isite, :] = x0
                    if verbose:
                        print("Inserting site {} at {}".format(isite, x0))
                    break

        self.nucleation_sites = sites.flatten().tolist()


    def move_fibers_to_surface(self, fibers: List[Fiber], ds_min: float, verbose: bool = True) -> None:
        """
        Take a list of fibers and randomly and uniformly place them normal to the surface with a minimum separation ds_min.

        Arguments
        ---------
        fibers : List[Fiber]
            List of fibers that will be moved. Only the Fiber.x property will be modified
        ds_min : float
            Minimum separation allowable between the fiber minus ends. Collisions are not searched for the rest of the fibers,
            though they are unlikely
        verbose : bool, default: :obj:`True`
            If true, print a progress message
        """
        print("Inserting fibers")
        for i in range(len(fibers)):
            (x0, u0) = self.find_binding_site(fibers[0:i], ds_min)
            if verbose:
                print("Inserting fiber {} at {}".format(i, x0))
            fibers[i].fill_node_positions(x0, u0)


@dataclass
class Point():
    """dataclass for a point force/torque source

    Attributes
    ----------
    position : List[float], default: :obj:`[0.0, 0.0, 0.0]`, units: :obj:`μm`
        Position of the point source (x,y,z)
    force : List[float], default: :obj:`[0.0, 0.0, 0.0]`, units: :obj:`pN`
        Constant force to emit from point source
    torque : List[float], default: :obj:`[0.0, 0.0, 0.0]`, units: :obj:`pN·μm`
        Constant torque to emit from point source
    time_to_live : float, default: :obj:`0.0`, units: :obj:`s`
        Simulation time after which the point source deactivates and does nothing. A value of
        0.0 means to live forever.
    """
    position: List[float] = field(default_factory=_default_vector)
    force: List[float] = field(default_factory=_default_vector)
    torque: List[float] = field(default_factory=_default_vector)
    time_to_live: float = 0.0


@dataclass
class BackgroundSource():
    """dataclass for a point force/torque source

    Attributes
    ----------
    components : List[int], default: :obj:`[0, 1, 2]`
        Component of position to scale (x,y,z)
        E.g. [1, 0, 2] to scale velocity in x by y, y by x, and z by z
    scale_factor : List[float], default: :obj:`[0.0, 0.0, 0.0]`, units: :obj:`1 / s`
        Amount to scale each component by
    uniform : List[float], default: :obj:`[0.0, 0.0, 0.0]`, units: :obj:`μm / s`
        Amount to scale each component by
    """
    components: List[int] = field(default_factory=_default_ivector)
    scale_factor: List[float] = field(default_factory=_default_vector)
    uniform: List[float] = field(default_factory=_default_vector)


@dataclass
class Config():
    """
    Parent dataclass for a SkellySim config. Use this config if you don't have a bounding volume

    Attributes
    ----------
    params : Params, default: :obj:`Params()`
        System parameters
    bodies : List[Body], default: :obj:`[]`
        List of bodies
    fibers : List[Fiber], default: :obj:`[]`
        List of fibers
    point_sources : List[Point], default: :obj:`[]`
        List of point sources
    """
    params: Params = field(default_factory=Params)
    bodies: List[Body] = field(default_factory=list)
    fibers: List[Fiber] = field(default_factory=list)
    point_sources: List[Point] = field(default_factory=list)
    background: BackgroundSource = field(default_factory=BackgroundSource)

    def plot_fibers(self):
        """
        Scatter plot fiber beginning and end points. Note axes are not scaled, so results may look
        'squished' and not exactly uniform.

        Returns
        -------
        tuple(matplotlib figure, matplotlib axis)
            position vector and its normalized version

        """
        import matplotlib.pyplot as plt
        x_fib = np.array([fib.x[0:3] for fib in self.fibers])
        x_fib_2 = np.array([fib.x[-3:] for fib in self.fibers])
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_fib[:, 0], x_fib[:, 1], x_fib[:, 2], color='blue')
        ax.scatter(x_fib_2[:, 0], x_fib_2[:, 1], x_fib_2[:, 2], color='green')
        plt.show()

    def save(self, filename: str = 'skelly_config.toml'):
        """
        Write config object to file in TOML format

        Arguments
        ---------
        filename : str, default: :obj:`skelly_config.toml`
            path of configuration file to output
        """
        check_type(self)
        if _check_invalid_attributes(self):
            print("Not saving configuration. Please fix listed attributes and try again")
            return

        with open(filename, 'w') as f:
            toml.dump(_unpack(self), f)


@dataclass
class ConfigSpherical(Config):
    """
    dataclass for a SkellySim config with a spherical periphery

    Attributes
    ----------
    params : Params, default: :obj:`Params()`
        System parameters
    bodies : List[Body], default: :obj:`[]`
        List of bodies
    fibers : List[Fiber], default: :obj:`[]`
        List of fibers
    point_sources : List[Point], default: :obj:`[]`
        List of point sources
    periphery : SphericalPeriphery, default: :obj:`SphericalPeriphery()`
        Spherical periphery object
    """
    periphery: SphericalPeriphery = field(default_factory=SphericalPeriphery)


@dataclass
class ConfigEllipsoidal(Config):
    """
    dataclass for a SkellySim config with an ellipsoidal periphery

    Attributes
    ----------
    params : Params, default: :obj:`Params()`
        System parameters
    bodies : List[Body], default: :obj:`[]`
        List of bodies
    fibers : List[Fiber], default: :obj:`[]`
        List of fibers
    point_sources : List[Point], default: :obj:`[]`
        List of point sources
    periphery : EllipsoidalPeriphery, default: :obj:`EllipsoidalPeriphery()`
        Periphery represented by an ellipsoid
    """
    periphery: EllipsoidalPeriphery = field(default_factory=EllipsoidalPeriphery)


@dataclass
class ConfigRevolution(Config):
    """
    dataclass for a SkellySim config with a 'surface of revolution' periphery

    Attributes
    ----------
    params : Params, default: :obj:`Params()`
        System parameters
    bodies : List[Body], default: :obj:`[]`
        List of bodies
    fibers : List[Fiber], default: :obj:`[]`
        List of fibers
    point_sources : List[Point], default: :obj:`[]`
        List of point sources
    periphery : RevolutionPeriphery, default: :obj:`RevolutionPeriphery()`
        Periphery represented by a surface of revolution
    """
    periphery: RevolutionPeriphery = field(default_factory=RevolutionPeriphery)
