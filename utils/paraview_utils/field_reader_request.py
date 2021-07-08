import msgpack
from pathlib import Path
from trajectory_utility import get_frame_info

outInfo = self.GetOutputInformation(0)
print("Initializing velocity field")
self.fhs, self.fpos, self.times = get_frame_info(sorted(Path('.').glob('skelly_sim.vf.*')))
print(self.times)
outInfo.Set(vtk.vtkStreamingDemandDrivenPipeline.TIME_RANGE(), [self.times[0], self.times[-1]], 2)
outInfo.Set(vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS(), self.times, len(self.times))
print("Finished init")
