import vtk
from pathlib import Path
from vtkmodules.vtkCommonDataModel import vtkDataSet, vtkPolyData, vtkMultiBlockDataSet
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain

class DesyncError(Exception):
    pass

def get_frame_info(filenames):
    import msgpack
    if not filenames:
        return [], [], []

    unpackers = []
    fhs = []
    times = []
    for filename in filenames:
        f = open(filename, "rb")
        fhs.append(f)
        unpackers.append(msgpack.Unpacker(f, raw=False))

    fpos = [[] for i in range(len(filenames))]
    while True:
        try:
            for i in range(len(unpackers)):
                fpos[i].append(unpackers[i].tell())
                if i == 0:
                    n_keys = unpackers[i].read_map_header()
                    for ikey in range(n_keys):
                        key = unpackers[i].unpack()
                        if key == 'time':
                            times.append(unpackers[i].unpack())
                        else:
                            unpackers[i].skip()
                else:
                    unpackers[i].skip()

        except msgpack.exceptions.OutOfData:
            fpos[0].pop()
            break

    return fhs, fpos, times


class SkellyReader(VTKPythonAlgorithmBase):
    def __init__(self, outputType, static=False, filepattern="skelly_sim.out.*"):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=0, nOutputPorts=1, outputType=outputType)

        import msgpack
        self.Unpacker = msgpack.Unpacker

        if static:
            self.times = None
        else:
            self.fhs, self.fpos, self.times = get_frame_info(sorted(Path('.').glob(filepattern)))

        # FIXME: Get this from user
        toml_file = sorted(Path('.').glob('*.toml'))[0]
        import toml
        with open(toml_file) as f:
            self.skelly_config = toml.load(f)

    def _load_frame(self, index, keys=("fibers", "bodies")):
        data = []
        for i in range(len(self.fhs)):
            self.fhs[i].seek(self.fpos[i][index])
            data.append(self.Unpacker(self.fhs[i], raw=False).unpack())

        time = data[0]["time"]
        dt = data[0]["dt"]
        if "fibers" in keys:
            fibers = []
            for el in data:
                if el["time"] != time or el["dt"] != dt:
                    raise DesyncError
                fibers.extend(el["fibers"][0])
                el.pop("fibers")

            data[0]["fibers"] = fibers

        if "bodies" in keys:
            data[0]["bodies"] = data[0]["bodies"][0]

        return data[0]

    def _get_timesteps(self):
        return self.times

    def _get_update_time(self, outInfo):
        executive = self.GetExecutive()
        timesteps = self._get_timesteps()
        if timesteps is None or len(timesteps) == 0:
            return None
        elif outInfo.Has(executive.UPDATE_TIME_STEP()) and len(timesteps) > 0:
            utime = outInfo.Get(executive.UPDATE_TIME_STEP())
            dtime = timesteps[0]
            for atime in timesteps:
                if atime > utime:
                    return dtime
                else:
                    dtime = atime
            return dtime
        else:
            assert(len(timesteps) > 0)
            return timesteps[0]

    def _get_update_timestep(self, outInfo):
        time = self._get_update_time(outInfo)

        timestep = len(self.times) - 1
        for i in range(len(self.times) - 1):
            if time < self.times[i+1] and time >= self.times[i]:
                timestep = i
                break
        return timestep

    def RequestInformation(self, request, inInfo, outInfo):
        executive = self.GetExecutive()
        info = outInfo.GetInformationObject(0)
        info.Remove(executive.TIME_STEPS())
        info.Remove(executive.TIME_RANGE())

        timesteps = self._get_timesteps()
        if timesteps is not None:
            for t in timesteps:
                info.Append(executive.TIME_STEPS(), t)
            info.Append(executive.TIME_RANGE(), timesteps[0])
            info.Append(executive.TIME_RANGE(), timesteps[-1])

        return 1


@smproxy.source(label="Fiber Reader")
class FiberReader(SkellyReader):
    def __init__(self):
        SkellyReader.__init__(self, outputType='vtkPolyData')

    @smproperty.doublevector(name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty")
    def GetTimestepValues(self):
        return self._get_timesteps()

    def RequestData(self, request, inInfo, outInfo):
        info = outInfo.GetInformationObject(0)
        timestep = self._get_update_timestep(info)
        frame = self._load_frame(timestep, keys=("fibers"))

        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        offset = 0
        for fib in frame["fibers"]:
            n_nodes = fib["n_nodes_"]
            lines.InsertNextCell(n_nodes)
            pl = vtk.vtkPolyLine()
            pl.GetPointIds().SetNumberOfIds(n_nodes)

            for i in range(n_nodes):
                low = 3 + i * 3
                lines.InsertCellPoint(offset)
                pts.InsertPoint(offset, fib["x_"][low : low + 3])
                offset += 1


        output = vtkPolyData.GetData(outInfo, 0)
        output.SetPoints(pts)
        output.SetLines(lines)

        return 1


@smproxy.source(label="Body Reader")
class BodyReader(SkellyReader):
    def __init__(self):
        SkellyReader.__init__(self, outputType='vtkMultiBlockDataSet')

    @smproperty.doublevector(name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty")
    def GetTimestepValues(self):
        return self._get_timesteps()

    def RequestData(self, request, inInfo, outInfo):
        info = outInfo.GetInformationObject(0)
        timestep = self._get_update_timestep(info)
        frame = self._load_frame(timestep, keys=("bodies"))

        mb = vtk.vtkMultiBlockDataSet()
        offset = 0
        for i, body in enumerate(frame["bodies"]):
            position = body["position_"][3:]
            s = vtk.vtkSphereSource()
            s.SetRadius(self.skelly_config["bodies"][i]['radius'])
            s.SetCenter(position)
            s.SetThetaResolution(32)
            s.SetPhiResolution(32)
            s.Update()
            mb.SetBlock(offset, s.GetOutput())
            offset += 1

        output = vtkMultiBlockDataSet.GetData(outInfo, 0)
        output.ShallowCopy(mb)

        return 1

@smproxy.source(label="Periphery Reader")
class PeripheryReader(SkellyReader):
    def __init__(self):
        SkellyReader.__init__(self, outputType='vtkPolyData', static=True)
        p = self.skelly_config['periphery']
        if p['shape'] == 'sphere':
            geom = vtk.vtkSphereSource()
            geom.SetRadius(p['radius'])
            geom.SetThetaResolution(32)
            geom.SetPhiResolution(32)
            geom.Update()
            self.source = geom.GetOutput()
        elif p['shape'] == 'ellipsoid':
            s = vtk.vtkParametricEllipsoid()
            s.SetXRadius(p['a'])
            s.SetYRadius(p['b'])
            s.SetZRadius(p['c'])

            geom = vtk.vtkParametricFunctionSource()
            geom.SetParametricFunction(s)
            geom.Update()
            self.source = geom.GetOutput()
        elif p['shape'] == 'surface_of_revolution':
            import numpy as np
            precompute_data = np.load(self.skelly_config['params']['shell_precompute_file'])
            # FIXME: Fixed node_scale_factor is no bueno. Reconstruct from shape_gallery? Read from precompute/config?
            nodes = precompute_data['nodes'] / 1.04
            nodes.reshape(nodes.size // 3, 3)

            from scipy.spatial import ConvexHull
            hull = ConvexHull(nodes)
            faces = vtk.vtkCellArray()
            points = vtk.vtkPoints()

            nodes = nodes.ravel()
            for i in range(nodes.size // 3):
                points.InsertPoint(i, nodes[3 * i : 3* (i + 1)])

            for face in hull.simplices:
                faces.InsertNextCell(3)
                for i in range(3):
                    faces.InsertCellPoint(face[i])

            self.source = vtkPolyData()
            self.source.SetPoints(points)
            self.source.SetPolys(faces)


    def RequestInformation(self, request, inInfo, outInfo):
        return 1

    def RequestData(self, request, inInfo, outInfo):
        output = vtk.vtkPolyData.GetData(outInfo, 0)
        output.ShallowCopy(self.source)

        return 1


@smproxy.source(label="Velocity Field Reader")
class SkellyFieldReader(SkellyReader):
    def __init__(self):
        SkellyReader.__init__(self, outputType='vtkPolyData', filepattern="skelly_sim.vf.*")

    @smproperty.doublevector(name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty")
    def GetTimestepValues(self):
        return self._get_timesteps()

    def _load_frame(self, index):
        data = []
        for i in range(len(self.fhs)):
            self.fhs[i].seek(self.fpos[i][index])
            data.append(self.Unpacker(self.fhs[i], raw=False).unpack())

        return data

    def RequestData(self, request, inInfo, outInfo):
        import numpy as np
        info = outInfo.GetInformationObject(0)
        timestep = self._get_update_timestep(info)
        frame = self._load_frame(timestep)

        pts = vtk.vtkPoints()
        npts = int(sum([data["x_grid"][2] for data in frame]))
        vertices = vtk.vtkPolyData()

        velocities = vtk.vtkDoubleArray()
        velocities.SetName("velocities")
        velocities.SetNumberOfComponents(3)
        velocities.SetNumberOfTuples(npts)

        magnitudes = vtk.vtkDoubleArray()
        magnitudes.SetName("magnitudes")
        magnitudes.SetNumberOfComponents(0)
        magnitudes.SetNumberOfValues(npts)

        offset = 0
        for data in frame:
            n_points_local = data["x_grid"][2]
            x_grid = data["x_grid"][3:]
            v_grid = data["v_grid"][3:]
            for i in range(n_points_local):
                pts.InsertPoint(offset, x_grid[3 * i : 3* (i + 1)])
                # verts.InsertNextCell(1, [offset])
                mag = np.linalg.norm(np.array(v_grid[3*i:3*(i+1)]))
                magnitudes.SetValue(offset, mag)
                velocities.SetTuple(offset, v_grid[3 * i : 3* (i + 1)])
                offset += 1


        output = vtkPolyData.GetData(outInfo, 0)
        output.SetPoints(pts)
        output.GetPointData().AddArray(velocities)
        output.GetPointData().AddArray(magnitudes)

        return 1
