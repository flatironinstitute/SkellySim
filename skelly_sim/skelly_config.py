from typing import List, Tuple
from dataclasses import dataclass, asdict, field, is_dataclass
from argparse import Namespace
import copy
import numpy as np
from scipy.special import ellipeinc, ellipe
from scipy.optimize import fsolve
import toml


def get_random_point_on_sphere():
    """
    Give a uniform random point on the surface of a unit sphere

    Returns: numpy 3d-vector of unit length on surface of a unit sphere
    """
    phi = np.random.uniform() * 2.0 * np.pi
    u = 2 * np.random.uniform() - 1.0
    factor = np.sqrt(1.0 - u * u)

    return np.array([np.cos(phi) * factor, np.sin(phi) * factor, u])


def sin_length(amplitude: float, xf: float):
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


def cos_length_full(amplitude: float, xi: float, xf: float, x_max: float):
    """
    The arclength of amplitude * cos(2*pi*x/x_max) on the interval [xi, xf]
    Designed to be used with an fsolve on (sin_length() - length) to get x_max
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
    return (ellipeinc(scale_factor * xf, -A2) -
            ellipeinc(scale_factor * xi, -A2)) / scale_factor


def get_random_orthogonal_vector(x: np.array):
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


def perturb_fiber(amplitude: float, length: float, x0: np.array, n_nodes: int):
    """
    Create a fiber with a small cosine perturbation in a random direction orthogonal to the fiber.
    Fiber orientation is assumed to be along the vector x0, which is the position of the minus end of the filament (so don't place at the origin).

    Arguments
    ---------
    amplitude : float
        Amplitude of the perturbation. Make it a small fraction of the total length
    length : float
        Length of the fiber
    x0 : np.array(3)
        3D position of the base of the Fiber. Don't place at the origin since the orientation is determined from this.
    n_nodes : int
        number of nodes to represent the fiber

    Returns: [n_nodes, 3] matrix of positions of each node in a perturbed fiber
    """
    x_max_fun = lambda xf: sin_length(amplitude, xf) - length
    x_max = fsolve(x_max_fun, length)

    normal = -x0 / np.linalg.norm(x0)

    ortho = get_random_orthogonal_vector(normal)

    arc_length_per_segment = length / (n_nodes - 1)
    lin_pos = np.zeros(n_nodes)
    for i in range(1, n_nodes):
        fun = lambda xf: cos_length_full(amplitude, lin_pos[i - 1], xf, x_max
                                         ) - arc_length_per_segment
        lin_pos[i] = fsolve(fun, lin_pos[i - 1] + arc_length_per_segment)

    fiber_positions = np.outer(lin_pos, normal)

    cos_perturbation = amplitude * (np.cos(lin_pos) - 1)
    fiber_positions += np.outer(cos_perturbation, ortho)
    fiber_positions += x0

    return fiber_positions


def unpack(obj):
    """
    Helper routine to process a Config recursively into the format needed by skelly_sim
    Used in toml dumping
    """
    if isinstance(obj, dict):
        return {key: unpack(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        if not obj:
            return
        return [unpack(value) for value in obj]
    elif is_dataclass(obj):
        return {key: unpack(value) for key, value in asdict(obj).items()}
    elif isinstance(obj, Namespace):
        return unpack(
            dict(
                filter(lambda keyval: not keyval[0].startswith('__'),
                       obj.__dict__.items())))
    elif isinstance(obj, tuple):
        return tuple(unpack(value) for value in obj)
    else:
        return obj


def default_vector():
    """
    The name says it all. A default vector for the field factories.
    """
    return [0.0, 0.0, 0.0]


def default_quaternion():
    """
    The name says it all. A default quaternion for the field factories.
    """
    return [0.0, 0.0, 0.0, 1.0]


@dataclass
class Fiber():
    """
    dataclass representing a single fiber

    Attributes
    ----------
    n_nodes : int
        Number of nodes to represent fiber. Highly deformed or very long fibers will require more nodes to be accurately represented
    parent_body : int
        Index of 'body' fiber is bound to. A value of '-1' indicates a free fiber
    force_scale : float
        Tangential force per unit length to act along filament. A positive value pushes toward the 'plus' end,
        while a negative value pushes toward the 'minus' end
    bending_rigidity : float
        Bending rigidity of this filament
    length : float
        Constraint length of this filament
    minus_clamped : bool
        Fix minus end of filament with "clamped" boundary condition, preserving orientation and position (Velocity = 0, AngularVelocity = 0)
    x : List[float]
        List of node positions in [x0,y0,z0,x1,y1,z1...] order. Extreme care must be taken when setting this since the length constraint
        can generate massive tensions with poor input
    """
    n_nodes: int = 32
    parent_body: int = -1
    force_scale: float = 0.0
    bending_rigidity: float = 2.5E-3
    length: float = 1.0
    minus_clamped: bool = False
    x: List[float] = field(default_factory=list)


@dataclass
class DynamicInstability():
    """
    dataclass for dynamic instability parameters

    Attributes
    ----------
    n_nodes : int
        Number of nodes for newly nucleated fibers (see nucleation_rate)
    v_growth : float
        Growth velocity in microns/second. Growth happens at the "plus" ends
    f_catastrophe : float
        Catastrophe frequency (probability per unit time of switching from growth to deletion) in 1/second
    v_grow_collision_scale : float
        When a fiber hits a boundary, scale its velocity (v_growth) by this factor
    f_catastrophe_collision_scale : float
        When a fiber hits a boundary, scale its catastrophe frequency (f_catastrophe) by this
    nucleation_rate : float
        Fiber nucleation rate in units of MT nucleations / second
    min_length: float
        New fiber initial length in microns
    bending_rigidity : float
        New fiber bending rigidity
    min_separation : float
        Minimum distance between Fiber minus ends in microns when nucleating (closer than this will be rejected)
    """
    n_nodes: int = 16
    v_growth: float = 0.0
    f_catastrophe: float = 0.0
    v_grow_collision_scale: float = 0.5
    f_catastrophe_collision_scale: float = 2.0
    nucleation_rate: float = 0.0
    min_length: float = 0.5
    bending_rigidity: float = 2.5E-3
    min_separation: float = 0.1


@dataclass
class VelocityField():
    """
    dataclass representing velocity field measurement parameters

    Attributes
    ----------
    moving_volume : bool
        Track velocity field around bodies. If two bodies are adjacent, their grids will be merged into one. Useful when no periphery.
    moving_volume_radius : float
        not really a radius. half box size for volume around body to track
    dt_write_field : float
        Time between velocity field measurements
    resolution : float
        Distance between grid points. n_points ~ (2 * radius / resolution)^3. Don't make too small unless you have lots of memory/storage :)
    """
    moving_volume: bool = True
    moving_volume_radius: float = 30.0
    dt_write_field: float = 0.5
    resolution: float = 1.0


@dataclass
class Params():
    """
    dataclass representing system/meta parameters for the entire simulation

    Attributes
    ----------
    eta : float
        Viscosity of fluid
    dt_initial : float
        Initial length of timestep in seconds
    dt_min : float
        Minimum timestep before simulation fails when using adaptive timestepping (adaptive_timestep_flag)
    dt_max : float
        Maximum timestep size allowable when using adaptive timestepping (adaptive_timestep_flag)
    dt_write : float
        Amount of simulation time between writes. Due to adaptive timestepping (and floating point issues) the
        time between writes is only approximately dt_write
    t_final : float
        Simulation time to quit the simulation
    gmres_tol : float
        GMRES tolerance, might be tuned later, but recommend at least 1E-8
    fiber_error_tol : float
        Fiber error tolerance. Not recommended to tamper with.
        Fiber error is the maximum deviation between 1.0 and a the derivative along the fiber.
        When using adaptive timestepping, if the error exceeds this value, the timestep is rejected.
    shell_precompute_file : str
        File to store the shell precompute data
    periphery_binding_flag : bool
        If set, fiber plus ends near the periphery (closer than 0.75, hardcoded) will use hinged boundary conditions
    seed : int
        Random number seed
    dynamic_instability : DynamicInstability
        Dynamic instability parameters
    velocity_field : VelocityField
        Velocity field parameters
    periphery_interaction_flag : bool
        Experimental repulsion between periphery and Fibers
    adaptive_timestep_flag : bool
        If set, use adaptive timestepping, which attempts to control simulation error by reducing the timestep when the solution has convergence issues
    """
    eta: float = 1.0
    dt_initial: float = 0.025
    dt_min: float = 1E-5
    dt_max: float = 0.025
    dt_write: float = 0.1
    t_final: float = 1000.0
    gmres_tol: float = 1E-8
    fiber_error_tol: float = 1E-1
    shell_precompute_file: str = "shell_precompute.npz"
    periphery_binding_flag: bool = False
    seed: int = 130319
    dynamic_instability: DynamicInstability = field(
        default_factory=DynamicInstability)
    velocity_field: VelocityField = field(default_factory=VelocityField)
    periphery_interaction_flag: bool = True
    adaptive_timestep_flag: bool = True


@dataclass
class Periphery():
    """
    dataclass representing a periphery. Don't use directly, but use one of the subclasses

    Attributes
    ----------
    n_nodes : int
        Number of nodes to represent Periphery object. larger peripheries = more nodes. Memory scales as n_nodes^2, so don't exceed ~10000
    """
    n_nodes: int = 6000

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
    n_nodes : int
        Number of nodes to represent Periphery object. larger peripheries = more nodes. Memory scales as n_nodes^2, so don't exceed ~10000
    shape : str
        Shape of the periphery. Don't modify it!
    radius : float
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
            u0 = get_random_point_on_sphere()
            x0 = 0.99999999 * u0 * self.radius

            accept = True
            for fib in fibers:
                if np.linalg.norm(x0 - fib.x[0:3]) < ds_min:
                    accept = False
                    break
            if accept:
                return (x0, u0)


@dataclass
class EllipsoidalPeriphery(Periphery):
    """
    dataclass representing an ellipsoidal periphery. (x/a)^2 + (y/b)^2 (z/c)^2 = 1

    Attributes
    ----------
    n_nodes : int
        Number of nodes to represent Periphery object. larger peripheries = more nodes. Memory scales as n_nodes^2, so don't exceed ~10000
    shape : str
        Shape of the periphery. Don't modify it!
    a : float
         length of axis 'a'
    b : float
         length of axis 'b'
    c : float
         length of axis 'c'
    """

    shape: str = 'ellipsoid'
    a: float = 7.8
    b: float = 4.16
    c: float = 4.16


@dataclass
class RevolutionPeriphery(Periphery):
    """
    dataclass representing a surface of revolution. The user provides an envelope function 'h(x)' (see skelly_sim.shape_gallery.Envelope)
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
        config.periphery.envelope.height = "0.5 * T * ((1 + 2*x/length)**p1) * ((1 - 2*x/length)**p2) * length"
        config.periphery.envelope.T = 0.72
        config.periphery.envelope.p1 = 0.4
        config.periphery.envelope.p2 = 0.2
        config.periphery.envelope.length = 7.5


    Attributes
    ----------
    n_nodes : int
        Number of nodes to represent Periphery object. This will be set by the envelope, so don't bother changing from default
    shape : str
        Shape of the periphery. Don't modify it!
    envelope : skelly_sim.shape_gallery.Envelope
        See example above
    """

    shape: str = 'surface_of_revolution'
    n_nodes: int = 0
    envelope: Namespace = field(default_factory=Namespace)


@dataclass
class Body():
    """
    dataclass for a single body and its parameters

    Attributes
    ----------
    nucleation_type : str
        How nucleation sites are made (just leave as 'auto', which will place the nucleation site at the fiber minus end automatically)
    n_nucleation_sites : int
        Number of available Fiber sites on the body. Don't add more fibers than this to body
    position : List[float]
        Lab frame coordinate of the body center
    orientation : List[float]
        Orientation quaternion of the body. Not worth changing
    shape : str
        Shape of the body. Sphere is currently only supported option
    radius : float
        Radius of the body. This is the attachment radius for nucleation sites, the hydrodynamic radius is a bit smaller
    num_nodes : int
        Number of nodes to represent surface. WARNING: MAKE NEW PRECOMPUTE DATA WHEN CHANGING or you will regret it.
    precompute_file : str
        Where precompute data is stored (quadrature data, mostly)
    external_force : List[float]
        Lab frame external force applied to body - useful for testing things like stokes flow
    """
    nucleation_type: str = 'auto'
    n_nucleation_sites: int = 500
    position: List[float] = field(default_factory=default_vector)
    orientation: List[float] = field(default_factory=default_quaternion)
    shape: str = 'sphere'
    radius: float = 1.0
    num_nodes: int = 600
    precompute_file: str = 'body_precompute.npz'
    external_force: List[float] = field(default_factory=default_vector)


@dataclass
class Point():
    """
    dataclass for a point force/torque source

    Attributes
    ----------
    position : List[float]
        Position of the point source (x,y,z)
    force : List[float]
        Constant force to emit from point source
    torque : List[float]
        Constant torque to emit from point source
    time_to_live : float
        Simulation time after which the point source deactivates and does nothing
    """
    position: List[float] = field(default_factory=default_vector)
    force: List[float] = field(default_factory=default_vector)
    torque: List[float] = field(default_factory=default_vector)
    time_to_live: float = 0.0


@dataclass
class Config():
    """
    Parent dataclass for a SkellySim config. Use this config if you don't have a bounding volume

    Attributes
    ----------
    params : Params
        System parameters
    bodies : List[Body]
        List of bodies
    fibers : List[Fiber]
        List of fibers
    point_sources : List[Point]
        List of point sources
    """
    params: Params = field(default_factory=Params)
    bodies: List[Body] = field(default_factory=list)
    fibers: List[Fiber] = field(default_factory=list)
    point_sources: List[Point] = field(default_factory=list)

    def insert_fiber(self, fiber: Fiber, placement=None):
        fib = copy.copy(fiber)
        if placement:
            if placement[0] == 'periphery':
                x0, u0 = self.periphery.find_binding_site(
                    self.fibers, placement[1])
            else:
                raise RuntimeError
            fib.x = (
                x0 -
                np.linspace(0, u0 * fib.length, fib.n_nodes)).ravel().tolist()
            fib.minus_clamped = True
        self.fibers.append(fib)

    def save(self, filename: str):
        with open(filename, 'w') as f:
            toml.dump(unpack(self), f)


@dataclass
class ConfigSpherical(Config):
    """
    dataclass for a SkellySim config with a spherical periphery

    Attributes
    ----------
    params : Params
        System parameters
    bodies : List[Body]
        List of bodies
    fibers : List[Fiber]
        List of fibers
    point_sources : List[Point]
        List of point sources
    periphery : SphericalPeriphery
        SphericalPeriphery
    """
    periphery: SphericalPeriphery = SphericalPeriphery()


@dataclass
class ConfigEllipsoidal(Config):
    """
    dataclass for a SkellySim config with an ellipsoidal periphery

    Attributes
    ----------
    params : Params
        System parameters
    bodies : List[Body]
        List of bodies
    fibers : List[Fiber]
        List of fibers
    point_sources : List[Point]
        List of point sources
    periphery : EllipsoidalPeriphery
        EllipsoidalPeriphery
    """
    periphery: EllipsoidalPeriphery = EllipsoidalPeriphery()


@dataclass
class ConfigRevolution(Config):
    """
    dataclass for a SkellySim config with a 'surface of revolution' periphery

    Attributes
    ----------
    params : Params
        System parameters
    bodies : List[Body]
        List of bodies
    fibers : List[Fiber]
        List of fibers
    point_sources : List[Point]
        List of point sources
    periphery : RevolutionPeriphery
        RevolutionPeriphery
    """
    periphery: RevolutionPeriphery = RevolutionPeriphery()
