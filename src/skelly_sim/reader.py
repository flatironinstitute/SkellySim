import numpy as np
import msgpack
import pickle
import toml
import os
import struct
from typing import List
from dataclasses import dataclass, field, asdict
from dataclass_utils import check_type
from subprocess import Popen, PIPE
from nptyping import NDArray, Shape, Float64

from skelly_sim.skelly_config import _check_invalid_attributes

def _ndencode(obj):
    if isinstance(obj, np.ndarray):
        return ['__eigen__', obj.shape[1], obj.shape[0]] + obj.ravel().tolist()
    else:
        return obj

def _default_3d_matrix():
    """
    A default matrix factory for dataclass 'field' objects: :obj:`[]`
    """
    return np.zeros(shape=(0, 3), dtype=np.float64)

def _eigen_to_numpy(d):
    """
    Recursively iterate through list/dict, converting lists that begin with '__eigen__' to
    appropriately sized numpy arrays.

    Arguments
    ---------
    d : Any
        object to convert

    Returns
    -------
    object with appropriately structured '__eigen__' lists replaced with numpy arrays
    """
    if isinstance(d, list):
        if d and d[0] == '__eigen__':
            if isinstance(d[1], float):
                # quaternion. not the cleanest, but eh
                return np.array(d[1:])
            if d[1] == 1 or d[2] == 1:
                # 1d array. just keep it that way
                return np.array(d[3:])
            elif d[1] == 3:
                # make our points along rows (along cols in c++, but c++ setup is colmajor, so they
                # cancel automatically)
                return np.array(d[3:]).reshape((d[2], d[1]))
            else:
                # keep same ordering of matrices as C++ otherwise
                return np.array(d[3:]).reshape((d[2], d[1])).transpose()
        else:
            return [_eigen_to_numpy(subd) for subd in d]
    elif isinstance(d, dict):
        for k, v in d.items():
            d[k] = _eigen_to_numpy(v)

    return d

@dataclass
class StreamlinesRequest:
    """Dataclass for requesting multiple streamlines

    Attributes
    ----------
    dt_init : float, default: :obj:`0.1`
        Initial timestep for streamline integrator (adaptive timestepper).
        Output points in streamline will be roughly dt_init separated in time
    t_final : float, default: :obj:`1.0`
        Final time to integrate to for streamline
    abs_err : float, default: :obj:`1E-10`
        Absolute tolerance in integrator. Lower will be more accurate, but take longer to evaluate
    rel_err : float, default: :obj:`1E-6`
        Relative tolerance in integrator. Lower will be more accurate, but take longer to evaluate
    back_integrate : bool, default: :obj:`True`
        Additionally integrate from [0, -t_final]
    x0 : NDArray[Shape["Any, 3"], Float64], default: :obj:`[]`, units: :obj:`μm`
        Position of the initial streamline seeds [[x0,y0,z0],[x1,y1,z1],...]
    """
    dt_init: float = 0.1
    t_final: float = 1.0
    abs_err: float = 1E-10
    rel_err: float = 1E-6
    back_integrate: bool = True
    x0: NDArray[Shape["Any, 3"], Float64] = field(default_factory=_default_3d_matrix)

@dataclass
class VelocityFieldRequest:
    """Dataclass for requesting a VelocityField

    Attributes
    ----------
    x0 : List[float], default: :obj:`[]`, units: :obj:`μm`
        Position of the field points [[x0,y0,z0],[x1,y1,z1],...]
    """
    x: NDArray[Shape["Any, 3"], Float64] = field(default_factory=_default_3d_matrix)

@dataclass
class Request:
    """Dataclass for a request to skelly_sim's listener functionality

    Attributes
    ----------
    frame_no : int, default: :obj:`0`
        Frame index of interest in trajectory
    evaluator : str, default: :obj:`CPU`
        Pair evaluator: "FMM", "CPU", or "GPU".
        CPU and GPU are typically fastest for smaller systems
    streamlines : StreamlinesRequest, default: :obj:`StreamlinesRequest()`
        Streamlines to build
    """
    frame_no: int = 0
    evaluator: str = "CPU"
    streamlines: StreamlinesRequest = field(default_factory=StreamlinesRequest)
    vortexlines: StreamlinesRequest = field(default_factory=StreamlinesRequest)
    velocity_field: VelocityFieldRequest = field(default_factory=VelocityFieldRequest)

class Listener:
    """
    Utility wrapper for interacting with the SkellySim binary directly for various post-processing tasks.
    Rather than interacting with stored binary data, this allows you to generate analysis data
    (such as streamlines and velocity fields) on the fly.

    Attributes
    ----------
    config_data : dict
        Global toml data associated with the simulation
    """

    def __init__(self, toml_file: str = 'skelly_config.toml', binary: str = 'skelly_sim'):
        """
        Initialize our TrajectoryReader object

        Arguments
        ---------
        toml_file : str
            Configuration file for the simulation. Usually 'skelly_config.toml', which is the default.
        """

        self.config_data: dict = {}

        with open(toml_file, 'r') as f:
            self.config_data = toml.load(f)

        self._proc = Popen([binary, '--listen'], stdin=PIPE, stdout=PIPE)


    def request(self, command: Request):
        check_type(command)
        if _check_invalid_attributes(self):
            print("Invalid request to Listener. Please fix listed attributes and try again")
            return None

        msg = msgpack.packb(asdict(command), default=_ndencode)

        self._proc.stdin.write(np.uint64(len(msg)))
        self._proc.stdin.write(msg)
        self._proc.stdin.flush()
        ressize = struct.unpack('<Q', self._proc.stdout.read(8))[0]
        if ressize:
            res = msgpack.unpackb(self._proc.stdout.read(ressize), object_hook=_eigen_to_numpy)
        else:
            res = None
        return res


    def __del__(self):
        self._proc.terminate()



class TrajectoryReader:
    """
    Utility wrapper for reading SkellySim trajectories. Provides a dict-like interface for access.

    Attributes
    ----------
    times : List[float]
        Simulation time of the corresponding frames
    config_data : dict
        Global toml data associated with the simulation
    """

    def __init__(self, toml_file: str = 'skelly_config.toml'):
        """
        Initialize our TrajectoryReader object

        Arguments
        ---------
        toml_file : str
            Configuration file for the simulation. Usually 'skelly_config.toml', which is the default.
        """

        self._fh = None
        self._fpos: List[int] = []
        self._frame_data: dict = None
        self.times: List[float] = []
        self.config_data: dict = {}

        with open(toml_file, 'r') as f:
            self.config_data = toml.load(f)

        traj_file = os.path.join(os.path.dirname(toml_file), 'skelly_sim.out')

        mtime = os.stat(traj_file).st_mtime
        index_file = traj_file + '.index'
        self._fh = open(traj_file, "rb")

        if os.path.isfile(index_file):
            with open(index_file, 'rb') as f:
                index = pickle.load(f)
                if index['mtime'] != mtime:
                    self._build_index(mtime, index_file)
                else:
                    self._fpos = index['fpos']
                    self.times = index['times']
        else:
            self._build_index(mtime, index_file)

    def load_frame(self, frameno: int):
        """
        Loads a trajectory frame into memory, replacing the last loaded frame

        If an invalid frame number is provided, then throws an IndexError.

        Arguments
        ---------
        frameno : int
            Frame of simulation to load. Valid values are [0, len(TrajectoryReader) - 1]
        """
        if abs(frameno) >= len(self):
            raise IndexError("Invalid frame number provided to TrajectoryReader")
        self._fh.seek(self._fpos[frameno])
        self._frame_data = msgpack.Unpacker(self._fh, raw=False, object_hook=_eigen_to_numpy).unpack()

    def _build_index(self, mtime: float, index_file: str):
        """
        Reads through the loaded trajectory, storing file position offsets and simulation times of each frame.
        Modifies self._fpos and self.times

        Arguments
        ---------
        mtime : float
            Output of os.stat(traj_file).st_mtime
        index_file : str
            Path to index file we wish to create
        """
        unpacker = msgpack.Unpacker(self._fh, raw=False)

        self._fpos = []
        self.times = []

        while True:
            try:
                self._fpos.append(unpacker.tell())
                n_keys = unpacker.read_map_header()
                for key in range(n_keys):
                    key = unpacker.unpack()
                    if key == 'time':
                        self.times.append(unpacker.unpack())
                    else:
                        unpacker.skip()

            except msgpack.exceptions.OutOfData:
                self._fpos.pop()
                break

        index = {
            'mtime': mtime,
            'fpos': self._fpos,
            'times': self.times,
        }
        with open(index_file, 'wb') as f:
            pickle.dump(index, f)

    def __getitem__(self, key):
        if key == 'bodies' or key == 'fibers':
            return self._frame_data[key][0]
        return self._frame_data[key]

    def __iter__(self):
        return iter(self._frame_data)

    def __len__(self):
        return len(self._fpos)

    def keys(self):
        return self._frame_data.keys()
