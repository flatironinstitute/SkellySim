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
        break

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

pd = self.GetPolyDataOutput()
pd.SetPoints(pts)
pd.SetLines(lines)
