from typing import List
from dataclasses import dataclass, asdict, field, is_dataclass
from scipy.special import ellipeinc, ellipe
from scipy.optimize import fsolve
import numpy as np
from argparse import Namespace


def unpack(obj):
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
    return [0.0, 0.0, 0.0]


def default_quaternion():
    return [0.0, 0.0, 0.0, 1.0]


@dataclass
class DynamicInstability():
    n_nodes: int = 16  # number of nodes representing fiber
    v_growth: float = 0.0  # growth velocity microns/second
    f_catastrophe: float = 0.0  # catastrophe frequency 1/second
    v_grow_collision_scale: float = 0.5  # when hit boundary, scale velocity by this
    f_catastrophe_collision_scale: float = 2.0  # when hit boundary, scale catastrophe freq by this
    nucleation_rate: float = 0.0  # fixed nucleation rate (MT nucleations / second)
    min_length: float = 0.5  # starting length in microns
    bending_rigidity: float = 20.0  # bending rigidity
    min_separation: float = 0.1  # minimum distance between MTs in microns when nucleating


@dataclass
class VelocityField():
    moving_volume: bool = True  # For large systems, track velocity field around bodies
    moving_volume_radius: float = 30.0  # not really a radius. half box size for volume around body to track
    dt_write_field: float = 0.5  # how often to write velocity field
    resolution: float = 1.0  # distance between grid points. n_points ~ (2 * radius / resolution)^3. don't make too small!


@dataclass
class Params():
    eta: float = 1.0  # Viscosity
    dt_initial: float = 0.025  # Initial length of timestep in seconds
    dt_min: float = 1E-5  # minimum timestep before simulation fails
    dt_max: float = 0.025  # maximum timestep size
    dt_write: float = 0.1  # roughly how often to write
    t_final: float = 1000.0  # time when finished
    gmres_tol: float = 1E-8  # gmres tolerance, might be tuned later
    fiber_error_tol: float = 1E-1  # don't touch, fiber error tolerance
    shell_precompute_file: str = "shell_precompute.npz"
    periphery_binding_flag: bool = False  # should MT hinge at periphery surface
    seed: int = 130319  # random number seed
    velocity_field_flag: bool = True  # measure the velocity field
    dynamic_instability: DynamicInstability = DynamicInstability()
    velocity_field: VelocityField = VelocityField()
    periphery_interaction_flag: bool = True
    adaptive_timestep_flag: bool = True

@dataclass
class Periphery():
    n_nodes: int = 6000  # number of nodes to represent sphere. larger peripheries = more nodes. don't exceed ~10000 :)


@dataclass
class SphericalPeriphery(Periphery):
    shape: str = 'sphere'  # fixed, don't change
    radius: float = 6.0  # radius

@dataclass
class EllipsoidalPeriphery(Periphery):
    shape: str = 'ellipsoid'  # fixed, don't change
    a: float = 7.8  # axis 1
    b: float = 4.16  # axis 2
    c: float = 4.16  # axis 3


@dataclass
class RevolutionPeriphery(Periphery):
    shape: str = 'surface_of_revolution'  # fixed, don't change
    n_nodes: int = 0
    envelope: Namespace = field(default_factory=Namespace)


@dataclass
class Body():
    nucleation_type: str = 'auto'  # how nucleation sites are made (can be manually set)
    n_nucleation_sites: int = 500  # number of available MT sites. don't add more fibers than this to body
    position: List[float] = field(
        default_factory=default_vector)  # lab frame coordinate
    orientation: List[float] = field(
        default_factory=default_quaternion
    )  # orientation quaternion. don't bother changing
    shape: str = 'sphere'  # sphere is currently only supported option
    radius: float = 1.0  # radius :). this is the attachment radius, the hydrodynamic radius is a bit smaller
    num_nodes: int = 600  # number of nodes to represent surface. MAKE NEW PRECOMPUTE DATA WHEN CHANGING
    precompute_file: str = 'body_precompute.npz'  # where precompute data is stored (quadrature data, mostly)
    external_force: List[float] = field(
        default_factory=default_vector)  # force applied to body


@dataclass
class Fiber():
    n_nodes: int = 32  # number of nodes
    parent_body: int = 0  # body it's attached to. if you add more bodies, you'll have to specify 0,1,2,3,...
    force_scale: float = -0.04  # tangential motor force on each node
    bending_rigidity: float = 0.1  # bending rigidity
    length: float = 10.0  # starting length
    # Flattened (x0,y0,z0,x1,y1,z1,...) absolute positions in real space of all the nodes, don't set unless you know what you're doing
    x: List[float] = field(default_factory=list)
    minus_clamped: bool = False


@dataclass
class Point():
    position: List[float] = field(default_factory=default_vector)
    force: List[float] = field(default_factory=default_vector)
    torque: List[float] = field(default_factory=default_vector)
    time_to_live: float = 0.0


@dataclass
class Config():
    params: Params = Params()
    bodies: List[Body] = field(default_factory=list)
    fibers: List[Fiber] = field(default_factory=list)
    point_sources: List[Point] = field(default_factory=list)


@dataclass
class ConfigSpherical(Config):
    periphery: SphericalPeriphery = SphericalPeriphery()


@dataclass
class ConfigEllipsoidal(Config):
    periphery: EllipsoidalPeriphery = EllipsoidalPeriphery()


@dataclass
class ConfigRevolution(Config):
    periphery: RevolutionPeriphery = RevolutionPeriphery()


def sin_length(amplitude: float, xf: float):
    A2 = (2 * np.pi * amplitude / xf)**2
    return xf / np.pi * (ellipe(-A2) + np.sqrt(1 + A2) * ellipe(A2 / (1 + A2)))


def cos_length_full(amplitude: float, xi: float, xf: float, x_max: float):
    scale_factor = 2.0 * np.pi / x_max
    A2 = (scale_factor * amplitude)**2
    return (ellipeinc(scale_factor * xf, -A2) -
            ellipeinc(scale_factor * xi, -A2)) / scale_factor


def get_random_orthogonal_vector(x: np.array):
    if x[1] != 0 or x[2] != 0:
        offaxis = np.array([1, 0, 0])
    else:
        offaxis = np.array([0, 1, 0])
    b = np.cross(x, offaxis)
    b /= np.linalg.norm(b)
    c = np.cross(x, b)

    theta = 2 * np.pi * np.random.uniform()
    return b * np.cos(theta) + c * np.sin(theta)

def get_random_point_on_sphere():
    phi = np.random.uniform() * 2.0 * np.pi
    u = 2 * np.random.uniform() - 1.0
    factor = np.sqrt(1.0 - u * u)

    return np.array([np.cos(phi) * factor, np.sin(phi) * factor, u])


def perturb_fiber(amplitude: float, length: float, x0: np.array, n_nodes: int):

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
