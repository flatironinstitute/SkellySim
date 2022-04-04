import numpy as np
import msgpack
import pickle
import toml
import os


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


class TrajectoryReader:
    """
    Utility wrapper for reading SkellySim trajectories. Provides a dict-like interface for access.

    Attributes
    ----------
    times : List[float]
        Simulation time of the corresponding frames
    config_data : dict
        Global toml data associated with the simulation
    
    Methods
    -------
    load_frame
        load frame into reader object by index of the trajectory frame. Raises IndexError if invalid index is provided
    """

    def __init__(self, toml_file: str = 'skelly_config.toml', velocity_field: bool = False):
        """
        Initialize our TrajectoryReader object

        Arguments
        ---------
        toml_file : str
            Configuration file for the simulation. Usually 'skelly_config.toml', which is the default.
        velocity_field : bool
            Set True to read the velocity field trajectory rather than the position trajectory
        """

        self._fh = None
        self._fpos: List[int] = []
        self._frame_data: dict = None
        self.times: List[float] = []
        self.config_data: dict = {}

        with open(toml_file, 'r') as f:
            self.config_data = toml.load(f)

        if velocity_field:
            traj_file = os.path.join(os.path.dirname(toml_file), 'skelly_sim.vf')
        else:
            traj_file = os.path.join(os.path.dirname(toml_file), 'skelly_sim.out')

        mtime = os.stat(traj_file).st_mtime
        index_file = traj_file + '.index'
        self._fh = open(traj_file, "rb")

        if os.path.isfile(index_file):
            with open(index_file, 'rb') as f:
                print("Loading trajectory index.")
                index = pickle.load(f)
                if index['mtime'] != mtime:
                    print("Stale trajectory index file. Rebuilding.")
                    self._build_index(mtime, index_file)
                else:
                    self._fpos = index['fpos']
                    self.times = index['times']
        else:
            print("No trajectory index file. Building.")
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
        if frameno < 0 or frameno >= len(self):
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
