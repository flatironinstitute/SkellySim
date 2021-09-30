import vtk
from pathlib import Path
from vtkmodules.vtkCommonDataModel import vtkDataSet, vtkPolyData, vtkMultiBlockDataSet
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain

class SkellyReader(VTKPythonAlgorithmBase):
    def __init__(self, outputType, static=False):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=0, nOutputPorts=1, outputType=outputType)

        if static:
            self.times = []
        else:
            from skelly_sim.paraview_utils.trajectory_utility import get_frame_info
            self.fhs, self.fpos, self.times = get_frame_info(sorted(Path('.').glob('skelly_sim.out.*')))

        # FIXME: Get this from user
        toml_file = sorted(Path('.').glob('*.toml'))[0]
        import toml
        with open(toml_file) as f:
            self.skelly_config = toml.load(f)

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
        from vtkmodules.vtkCommonDataModel import vtkPolyData
        from skelly_sim.paraview_utils.trajectory_utility import load_frame
        info = outInfo.GetInformationObject(0)

        timestep = self._get_update_timestep(info)

        frame = load_frame(self.fhs, self.fpos, timestep)

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
        from skelly_sim.paraview_utils.trajectory_utility import load_frame
        info = outInfo.GetInformationObject(0)

        timestep = self._get_update_timestep(info)

        frame = load_frame(self.fhs, self.fpos, timestep)

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
            self.source = vtk.vtkSphereSource()
            self.source.SetRadius(p['radius'])
            self.source.SetThetaResolution(32)
            self.source.SetPhiResolution(32)
        elif p['shape'] == 'ellipsoid':
            s = vtk.vtkParametricEllipsoid()
            s.SetXRadius(p['a'])
            s.SetYRadius(p['b'])
            s.SetZRadius(p['c'])

            self.source = vtk.vtkParametricFunctionSource()
            self.source.SetParametricFunction(s)

    def RequestInformation(self, request, inInfo, outInfo):
        return 1

    def RequestData(self, request, inInfo, outInfo):
        self.source.Update()
        output = vtk.vtkPolyData.GetData(outInfo, 0)
        output.ShallowCopy(self.source.GetOutput())

        return 1
