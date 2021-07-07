import vtk
from trajectory_utility import load_frame

outInfo = self.GetOutputInformation(0)

if outInfo.Has(vtk.vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP()):
    time = outInfo.Get(vtk.vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP())
else:
    time = 0


timestep = len(self.times) - 1
for i in range(len(self.times) - 1):
    if time < self.times[i+1] and time >= self.times[i]:
        timestep = i

frame = load_frame(self.fhs, self.fpos, timestep)

mb = vtk.vtkMultiBlockDataSet()
offset = 0
for body in frame["bodies"]:
    position = body["position_"][3:]
    s = vtk.vtkSphereSource()
    s.SetRadius(0.5)
    s.SetCenter(position)
    s.SetThetaResolution(32)
    s.SetPhiResolution(32)
    s.Update()
    mb.SetBlock(offset, s.GetOutput())
    offset += 1

self.GetOutput().ShallowCopy(mb)
