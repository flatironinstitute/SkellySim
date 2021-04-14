import msgpack
from pathlib import Path
from trajectory_utility import get_frame_sizes

outInfo = self.GetOutputInformation(0)
print("Initializing fibers")
self.fhs, self.fpos = get_frame_sizes(sorted(Path('.').glob('skelly_sim.out.*')))
timesteps = range(len(self.fpos[0]))
outInfo.Set(vtk.vtkStreamingDemandDrivenPipeline.TIME_RANGE(), [timesteps[0], timesteps[-1]], 2)
outInfo.Set(vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS(), timesteps, len(timesteps))
