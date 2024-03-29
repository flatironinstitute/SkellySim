import numpy as np
import msgpack
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
    A default matrix factory for dataclass 'field' objects: :obj:`np.zeros(shape=(0, 3))`
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
        elif d and d[0] == '__quat__':
            # Quaternion, return the array
            return np.array(d[1:])
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
    x0 : NDArray[Shape["Any, 3"], Float64], default: :obj:`np.zeros(shape=(0, 3))`, units: :obj:`μm`
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
    x : NDArray[Shape["Any, 3"], Float64], default: :obj:`np.zeros(shape=(0, 3))`, units: :obj:`μm`
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
    vortexlines : StreamlinesRequest, default: :obj:`StreamlinesRequest()`
        Vortex lines to build
    velocity_field : VelocityFieldRequest, default :obj:`VelocityFieldRequest()`
        Velocity field to build
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
        toml_file : str, default: 'skelly_config.toml'
            Configuration file for the simulation. Usually 'skelly_config.toml', which is the default.
        binary : str, default: 'skelly_sim'
            Name of SkellySim executable in your PATH or full path to SkellySim executable
        """

        self.config_data: dict = {}

        with open(toml_file, 'r') as f:
            self.config_data = toml.load(f)

        self._proc = Popen(f'mpirun -n 1 -q -mca orte_abort_on_non_zero_status false {binary} --listen'.split(' '), stdin=PIPE, stdout=PIPE)


    def request(self, command: Request):
        """
        Execute request on listener subprocess

        Arguments
        ---------
        command : Request
            Request payload

        Returns
        -------
            Dictionary of result data. Should be relatively self-documenting. Check examples or res.keys().
        """
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
        self._proc.stdin.write(np.uint64(0))
        try: 
            self._proc.terminate()
        except:
            pass



class TrajectoryReader:
    """
    Utility wrapper for reading SkellySim trajectories. Provides a dict-like interface for access.

    Attributes
    ----------
    times : List[float]
        Simulation time of the corresponding frames
    config_data : dict
        Global toml data associated with the simulation
    fiber_type : int
        Fiber type associated with the simulation
    trajectory_version : int
        SkellySim trajectory version (if defined, otherwise, 0)
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
        self._unpacker = None               # Internal unpacker for the header and index routines
        self._fpos: List[int] = []
        self._frame_data: dict = None
        self.times: List[float] = []
        self.config_data: dict = {}
        self.header_data: dict = {}
        self.fiber_type: int = 0
        self.trajectory_version: int = 0

        with open(toml_file, 'r') as f:
            self.config_data = toml.load(f)

        traj_file = os.path.join(os.path.dirname(toml_file), 'skelly_sim.out')

        mtime = int(os.stat(traj_file).st_mtime)
        index_file = traj_file + '.cindex'
        self._fh = open(traj_file, "rb")

        # Try to unpack the first object in the file, as it may be a header, or might not be
        self._read_header()

        if os.path.isfile(index_file):
            with open(index_file, 'rb') as f:
                index = msgpack.load(f)
                if index['mtime'] != mtime or 'times' not in index:
                    self._build_index(mtime, index_file)
                else:
                    self._fpos = index['offsets']
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

    def _read_header(self):
        """
        Read the header (if it exists) in the file, or detect if we cannot do this, to make sur ethe file
        is set appropriately.
        """
        self._unpacker = msgpack.Unpacker(self._fh, raw=False)
        header_data = next(self._unpacker)
        # Check to see if we have old data, or new data (does the header exist)
        if 'trajversion' in header_data:
            # Good news, we have a header!
            self.header_data = header_data
            self.trajectory_version = self.header_data['trajversion']
            self.fiber_type = self.header_data['fiber_type']
        else:
            # Bad news, we do not have a header!
            self.trajectory_version = 0
            self.fiber_type = 0
            # Reset the file pointer for the _build_index routine
            self._fh.seek(0)

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
        self._fpos = []
        self.times = []

        while True:
            try:
                self._fpos.append(self._unpacker.tell())
                n_keys = self._unpacker.read_map_header()
                for key in range(n_keys):
                    key = self._unpacker.unpack()
                    if key == 'time':
                        self.times.append(self._unpacker.unpack())
                    else:
                        self._unpacker.skip()

            except msgpack.exceptions.OutOfData:
                self._fpos.pop()
                break

        index = {
            'mtime': mtime,
            'offsets': self._fpos,
            'times': self.times,
        }
        with open(index_file, 'wb') as f:
            msgpack.dump(index, f)

    def __getitem__(self, key):
        if self.trajectory_version < 1:
            if key == 'bodies' or key == 'fibers':
                return self._frame_data[key][0]
            return self._frame_data[key][0]
        else:
            if key == 'bodies':
                # There are number types of bodies, flatten them (since we are in python and heterogeneous types
                # aren't a problem) and return
                flattened_bodies = [item for sublist in self._frame_data[key] for item in sublist]
                return flattened_bodies
            elif key == 'fibers':
                # Fibers are kept one level below due to abstraction
                return self._frame_data[key][1]
            return self._frame_data[key]

    def __iter__(self):
        return iter(self._frame_data)

    def __len__(self):
        return len(self._fpos)

    def keys(self):
        return self._frame_data.keys()
