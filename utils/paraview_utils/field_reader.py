import vtk
from trajectory_utility import load_field_frame
import numpy as np

outInfo = self.GetOutputInformation(0)

if outInfo.Has(vtk.vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP()):
    time = outInfo.Get(vtk.vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP())
else:
    time = 0

timestep = len(self.times) - 1
for i in range(len(self.times) - 1):
    if time < self.times[i+1] and time >= self.times[i]:
        timestep = i

frame = load_field_frame(self.fhs, self.fpos, timestep)

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

pd = self.GetPolyDataOutput()
pd.SetPoints(pts)
pd.GetPointData().AddArray(velocities)
pd.GetPointData().AddArray(magnitudes)
