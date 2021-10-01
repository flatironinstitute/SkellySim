# state file generated using paraview version 5.9.1

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1390, 924]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [0.008008480072021484, 0.0, 0.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [0.008008480072021484, -23.035984688253333, 0.0]
renderView1.CameraFocalPoint = [0.008008480072021484, 0.0, 0.0]
renderView1.CameraViewUp = [0.0, 0.0, 1.0]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 12.780401367561524

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1390, 924)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'Skelly Bodies'
bodies = SkellyBodies(registrationName='Bodies')

# show data from bodies
bodiesDisplay = Show(bodies, renderView1, 'GeometryRepresentation')

# create a new 'Skelly Velocity Field'
velocityField = SkellyVelocityField(registrationName='Velocity Field')

# show data from velocityField
velocityFieldDisplay = Show(velocityField, renderView1, 'GeometryRepresentation')

# create a new 'Skelly Fibers'
fibers = SkellyFibers(registrationName='Fibers')

# show data from fibers
fibersDisplay = Show(fibers, renderView1, 'GeometryRepresentation')

# create a new 'Tube'
fiberTubes = Tube(registrationName='Fiber Tubes', Input=fibers)
fiberTubes.Scalars = ['POINTS', '']
fiberTubes.Vectors = ['POINTS', '1']
fiberTubes.NumberofSides = 10
fiberTubes.Radius = 0.0125

# create a new 'Glyph'
arrows = Glyph(registrationName='Arrows', Input=velocityField,
    GlyphType='Arrow')
arrows.OrientationArray = ['POINTS', 'velocities']
arrows.ScaleArray = ['POINTS', 'No scale array']
arrows.ScaleFactor = 1.4000000000000001
arrows.GlyphTransform = 'Transform2'

# create a new 'Skelly Periphery'
periphery = SkellyPeriphery(registrationName='Periphery')

# show data from periphery
peripheryDisplay = Show(periphery, renderView1, 'GeometryRepresentation')

try:
    # create a new 'Decimate'
    peripheryDecimator = Decimate(registrationName='Periphery Decimator', Input=periphery)
    peripheryDecimator.TargetReduction = 0.86
except:
    pass

# create a new 'Clip Closed Surface'
clipPeriphery = ClipClosedSurface(registrationName='Clip Periphery', Input=periphery)
clipPeriphery.ClippingPlane = 'Plane'

# init the 'Plane' selected for 'ClippingPlane'
clipPeriphery.ClippingPlane.Origin = [0.008008718490600586, 0.0, 0.0]
clipPeriphery.ClippingPlane.Normal = [0.0, 1.0, 0.0]

# create a new 'Point Volume Interpolator'
pointVolumeInterpolator = PointVolumeInterpolator(registrationName='Point-Volume Interpolator', Input=velocityField,
    Source='Bounded Volume')
pointVolumeInterpolator.Kernel = 'VoronoiKernel'
pointVolumeInterpolator.NullPointsStrategy = 'Mask Points'
pointVolumeInterpolator.Locator = 'Static Point Locator'

# init the 'Bounded Volume' selected for 'Source'
pointVolumeInterpolator.Source.Origin = [-6.800000190734863, -3.1600000858306885, -3.1600000858306885]
pointVolumeInterpolator.Source.Scale = [14.0, 7.0, 7.0]

try:
    # create a new 'Stream Tracer With Custom Source'
    streamlineFromPeriphery = StreamTracerWithCustomSource(registrationName='Streamline From Periphery', Input=pointVolumeInterpolator,
        SeedSource=peripheryDecimator)
    streamlineFromPeriphery.Vectors = ['POINTS', 'velocities']
    streamlineFromPeriphery.MaximumStreamlineLength = 14.000000000000002

    # create a new 'Tube'
    streamlinePeripheryTubes = Tube(registrationName='Streamline Periphery Tubes', Input=streamlineFromPeriphery)
    # show data from streamlinePeripheryTubes
    streamlinePeripheryTubesDisplay = Show(streamlinePeripheryTubes, renderView1, 'GeometryRepresentation')

    streamlinePeripheryTubes.Scalars = ['POINTS', '']
    streamlinePeripheryTubes.Vectors = ['POINTS', 'Normals']
    streamlinePeripheryTubes.NumberofSides = 10
    streamlinePeripheryTubes.Radius = 0.05

    # create a new 'Glyph'
    streamlinePeripheryArrows = Glyph(registrationName='Streamline Periphery Arrows', Input=streamlineFromPeriphery,
        GlyphType='Arrow')
    # show data from streamlinePeripheryArrows
    streamlinePeripheryArrowsDisplay = Show(streamlinePeripheryArrows, renderView1, 'GeometryRepresentation')
    streamlinePeripheryArrows.OrientationArray = ['POINTS', 'velocities']
    streamlinePeripheryArrows.ScaleArray = ['POINTS', 'No scale array']
    streamlinePeripheryArrows.ScaleFactor = 1.400212049484253
    streamlinePeripheryArrows.GlyphTransform = 'Transform2'
    streamlinePeripheryArrows.GlyphMode = 'Every Nth Point'
    streamlinePeripheryArrows.Stride = 20
except:
    pass

# create a new 'Stream Tracer'
streamlines = StreamTracer(registrationName='Streamlines', Input=pointVolumeInterpolator,
    SeedType='Point Cloud')
# show data from streamlines
streamlinesDisplay = Show(streamlines, renderView1, 'GeometryRepresentation')
streamlines.Vectors = ['POINTS', 'velocities']
streamlines.MaximumStreamlineLength = 14.000000000000002

# init the 'Point Cloud' selected for 'SeedType'
streamlines.SeedType.Center = [0.1999998092651376, 0.33999991416931197, 0.33999991416931197]
streamlines.SeedType.Radius = 1.4000000000000004

# create a new 'Tube'
streamlineTubes = Tube(registrationName='Streamline Tubes', Input=streamlines)
# show data from streamlineTubes
streamlineTubesDisplay = Show(streamlineTubes, renderView1, 'GeometryRepresentation')
streamlineTubes.Scalars = ['POINTS', '']
streamlineTubes.Vectors = ['POINTS', 'Normals']
streamlineTubes.Radius = 0.05

# create a new 'Glyph'
streamlineArrows = Glyph(registrationName='Streamline Arrows', Input=streamlines,
    GlyphType='Arrow')
# show data from streamlineArrows
streamlineArrowsDisplay = Show(streamlineArrows, renderView1, 'GeometryRepresentation')
streamlineArrows.OrientationArray = ['POINTS', 'velocities']
streamlineArrows.ScaleArray = ['POINTS', 'No scale array']
streamlineArrows.GlyphTransform = 'Transform2'
streamlineArrows.GlyphMode = 'Every Nth Point'
streamlineArrows.Stride = 20

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# trace defaults for the display properties.
peripheryDisplay.Representation = 'Surface'
peripheryDisplay.AmbientColor = [0.9137254901960784, 0.9137254901960784, 0.9137254901960784]
peripheryDisplay.ColorArrayName = ['POINTS', '']
peripheryDisplay.DiffuseColor = [0.9137254901960784, 0.9137254901960784, 0.9137254901960784]
peripheryDisplay.Opacity = 0.1
peripheryDisplay.SelectTCoordArray = 'None'
peripheryDisplay.SelectNormalArray = 'Normals'
peripheryDisplay.SelectTangentArray = 'None'
peripheryDisplay.OSPRayScaleArray = 'Normals'
peripheryDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
peripheryDisplay.SelectOrientationVectors = 'None'
peripheryDisplay.ScaleFactor = 1.5575967311859131
peripheryDisplay.SelectScaleArray = 'None'
peripheryDisplay.GlyphType = 'Arrow'
peripheryDisplay.GlyphTableIndexArray = 'None'
peripheryDisplay.GaussianRadius = 0.07787983655929566
peripheryDisplay.SetScaleArray = ['POINTS', 'Normals']
peripheryDisplay.ScaleTransferFunction = 'PiecewiseFunction'
peripheryDisplay.OpacityArray = ['POINTS', 'Normals']
peripheryDisplay.OpacityTransferFunction = 'PiecewiseFunction'
peripheryDisplay.DataAxesGrid = 'GridAxesRepresentation'
peripheryDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
peripheryDisplay.ScaleTransferFunction.Points = [-17.252302169799805, 0.0, 0.5, 0.0, 17.287822723388672, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
peripheryDisplay.OpacityTransferFunction.Points = [-17.252302169799805, 0.0, 0.5, 0.0, 17.287822723388672, 1.0, 0.5, 0.0]

# trace defaults for the display properties.
bodiesDisplay.Representation = 'Surface'
bodiesDisplay.AmbientColor = [0.6666666666666666, 1.0, 1.0]
bodiesDisplay.ColorArrayName = ['POINTS', '']
bodiesDisplay.DiffuseColor = [0.6666666666666666, 1.0, 1.0]
bodiesDisplay.SelectTCoordArray = 'None'
bodiesDisplay.SelectNormalArray = 'None'
bodiesDisplay.SelectTangentArray = 'None'
bodiesDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
bodiesDisplay.SelectOrientationVectors = 'None'
bodiesDisplay.ScaleFactor = -2.0000000000000002e+298
bodiesDisplay.SelectScaleArray = 'None'
bodiesDisplay.GlyphType = 'Arrow'
bodiesDisplay.GlyphTableIndexArray = 'None'
bodiesDisplay.GaussianRadius = 1.0
bodiesDisplay.SetScaleArray = ['POINTS', '']
bodiesDisplay.ScaleTransferFunction = 'PiecewiseFunction'
bodiesDisplay.OpacityArray = ['POINTS', '']
bodiesDisplay.OpacityTransferFunction = 'PiecewiseFunction'
bodiesDisplay.DataAxesGrid = 'GridAxesRepresentation'
bodiesDisplay.PolarAxes = 'PolarAxesRepresentation'

# trace defaults for the display properties.
velocityFieldDisplay.Representation = 'Surface'
velocityFieldDisplay.ColorArrayName = ['POINTS', '']
velocityFieldDisplay.SelectTCoordArray = 'None'
velocityFieldDisplay.SelectNormalArray = 'None'
velocityFieldDisplay.SelectTangentArray = 'None'
velocityFieldDisplay.OSPRayScaleArray = 'magnitudes'
velocityFieldDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
velocityFieldDisplay.SelectOrientationVectors = 'None'
velocityFieldDisplay.ScaleFactor = 1.4000000000000001
velocityFieldDisplay.SelectScaleArray = 'None'
velocityFieldDisplay.GlyphType = 'Arrow'
velocityFieldDisplay.GlyphTableIndexArray = 'None'
velocityFieldDisplay.GaussianRadius = 0.07
velocityFieldDisplay.SetScaleArray = ['POINTS', 'magnitudes']
velocityFieldDisplay.ScaleTransferFunction = 'PiecewiseFunction'
velocityFieldDisplay.OpacityArray = ['POINTS', 'magnitudes']
velocityFieldDisplay.OpacityTransferFunction = 'PiecewiseFunction'
velocityFieldDisplay.DataAxesGrid = 'GridAxesRepresentation'
velocityFieldDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
velocityFieldDisplay.ScaleTransferFunction.Points = [0.02904904104653732, 0.0, 0.5, 0.0, 0.23994956895932976, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
velocityFieldDisplay.OpacityTransferFunction.Points = [0.02904904104653732, 0.0, 0.5, 0.0, 0.23994956895932976, 1.0, 0.5, 0.0]

# get color transfer function/color map for 'velocities'
velocitiesLUT = GetColorTransferFunction('velocities')
velocitiesLUT.RGBPoints = [0.0040314331047620125, 0.231373, 0.298039, 0.752941, 0.23042085554452357, 0.865003, 0.865003, 0.865003, 0.45681027798428525, 0.705882, 0.0156863, 0.14902]
velocitiesLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
streamlinePeripheryTubesDisplay.Representation = 'Surface'
streamlinePeripheryTubesDisplay.ColorArrayName = ['POINTS', 'velocities']
streamlinePeripheryTubesDisplay.LookupTable = velocitiesLUT
streamlinePeripheryTubesDisplay.SelectTCoordArray = 'None'
streamlinePeripheryTubesDisplay.SelectNormalArray = 'None'
streamlinePeripheryTubesDisplay.SelectTangentArray = 'None'
streamlinePeripheryTubesDisplay.OSPRayScaleArray = 'AngularVelocity'
streamlinePeripheryTubesDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
streamlinePeripheryTubesDisplay.SelectOrientationVectors = 'None'
streamlinePeripheryTubesDisplay.ScaleFactor = 1.4022116661071777
streamlinePeripheryTubesDisplay.SelectScaleArray = 'None'
streamlinePeripheryTubesDisplay.GlyphType = 'Arrow'
streamlinePeripheryTubesDisplay.GlyphTableIndexArray = 'None'
streamlinePeripheryTubesDisplay.GaussianRadius = 1.0
streamlinePeripheryTubesDisplay.SetScaleArray = ['POINTS', '']
streamlinePeripheryTubesDisplay.ScaleTransferFunction = 'PiecewiseFunction'
streamlinePeripheryTubesDisplay.OpacityArray = ['POINTS', '']
streamlinePeripheryTubesDisplay.OpacityTransferFunction = 'PiecewiseFunction'
streamlinePeripheryTubesDisplay.DataAxesGrid = 'GridAxesRepresentation'
streamlinePeripheryTubesDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
streamlinePeripheryTubesDisplay.ScaleTransferFunction.Points = [-0.6383726359147425, 0.0, 0.5, 0.0, 0.7904715711954985, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
streamlinePeripheryTubesDisplay.OpacityTransferFunction.Points = [-0.6383726359147425, 0.0, 0.5, 0.0, 0.7904715711954985, 1.0, 0.5, 0.0]

# trace defaults for the display properties.
streamlinePeripheryArrowsDisplay.Representation = 'Surface'
streamlinePeripheryArrowsDisplay.ColorArrayName = ['POINTS', 'velocities']
streamlinePeripheryArrowsDisplay.LookupTable = velocitiesLUT
streamlinePeripheryArrowsDisplay.SelectTCoordArray = 'None'
streamlinePeripheryArrowsDisplay.SelectNormalArray = 'None'
streamlinePeripheryArrowsDisplay.SelectTangentArray = 'None'
streamlinePeripheryArrowsDisplay.OSPRayScaleArray = 'AngularVelocity'
streamlinePeripheryArrowsDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
streamlinePeripheryArrowsDisplay.SelectOrientationVectors = 'None'
streamlinePeripheryArrowsDisplay.ScaleFactor = 1.3969835281372072
streamlinePeripheryArrowsDisplay.SelectScaleArray = 'None'
streamlinePeripheryArrowsDisplay.GlyphType = 'Arrow'
streamlinePeripheryArrowsDisplay.GlyphTableIndexArray = 'None'
streamlinePeripheryArrowsDisplay.GaussianRadius = 1.0
streamlinePeripheryArrowsDisplay.SetScaleArray = ['POINTS', '']
streamlinePeripheryArrowsDisplay.ScaleTransferFunction = 'PiecewiseFunction'
streamlinePeripheryArrowsDisplay.OpacityArray = ['POINTS', '']
streamlinePeripheryArrowsDisplay.OpacityTransferFunction = 'PiecewiseFunction'
streamlinePeripheryArrowsDisplay.DataAxesGrid = 'GridAxesRepresentation'
streamlinePeripheryArrowsDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
streamlinePeripheryArrowsDisplay.ScaleTransferFunction.Points = [-0.6181262309319528, 0.0, 0.5, 0.0, 0.7904715360928054, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
streamlinePeripheryArrowsDisplay.OpacityTransferFunction.Points = [-0.6181262309319528, 0.0, 0.5, 0.0, 0.7904715360928054, 1.0, 0.5, 0.0]

# trace defaults for the display properties.
fibersDisplay.Representation = 'Surface'
fibersDisplay.ColorArrayName = ['POINTS', '']
fibersDisplay.SelectTCoordArray = 'None'
fibersDisplay.SelectNormalArray = 'None'
fibersDisplay.SelectTangentArray = 'None'
fibersDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
fibersDisplay.SelectOrientationVectors = 'None'
fibersDisplay.ScaleFactor = 1.497609043121338
fibersDisplay.SelectScaleArray = 'None'
fibersDisplay.GlyphType = 'Arrow'
fibersDisplay.GlyphTableIndexArray = 'None'
fibersDisplay.GaussianRadius = 0.07488048791885377
fibersDisplay.SetScaleArray = ['POINTS', '']
fibersDisplay.ScaleTransferFunction = 'PiecewiseFunction'
fibersDisplay.OpacityArray = ['POINTS', '']
fibersDisplay.OpacityTransferFunction = 'PiecewiseFunction'
fibersDisplay.DataAxesGrid = 'GridAxesRepresentation'
fibersDisplay.PolarAxes = 'PolarAxesRepresentation'

# trace defaults for the display properties.
streamlinesDisplay.Representation = 'Surface'
streamlinesDisplay.ColorArrayName = ['POINTS', '']
streamlinesDisplay.SelectTCoordArray = 'None'
streamlinesDisplay.SelectNormalArray = 'None'
streamlinesDisplay.SelectTangentArray = 'None'
streamlinesDisplay.OSPRayScaleArray = 'AngularVelocity'
streamlinesDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
streamlinesDisplay.SelectOrientationVectors = 'Normals'
streamlinesDisplay.ScaleFactor = 1.4001784324645996
streamlinesDisplay.SelectScaleArray = 'AngularVelocity'
streamlinesDisplay.GlyphType = 'Arrow'
streamlinesDisplay.GlyphTableIndexArray = 'AngularVelocity'
streamlinesDisplay.GaussianRadius = 0.07000892162322998
streamlinesDisplay.SetScaleArray = ['POINTS', 'AngularVelocity']
streamlinesDisplay.ScaleTransferFunction = 'PiecewiseFunction'
streamlinesDisplay.OpacityArray = ['POINTS', 'AngularVelocity']
streamlinesDisplay.OpacityTransferFunction = 'PiecewiseFunction'
streamlinesDisplay.DataAxesGrid = 'GridAxesRepresentation'
streamlinesDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
streamlinesDisplay.ScaleTransferFunction.Points = [-0.4927095146685523, 0.0, 0.5, 0.0, 0.6623530655842629, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
streamlinesDisplay.OpacityTransferFunction.Points = [-0.4927095146685523, 0.0, 0.5, 0.0, 0.6623530655842629, 1.0, 0.5, 0.0]

# trace defaults for the display properties.
streamlineTubesDisplay.Representation = 'Surface'
streamlineTubesDisplay.ColorArrayName = ['POINTS', 'velocities']
streamlineTubesDisplay.LookupTable = velocitiesLUT
streamlineTubesDisplay.SelectTCoordArray = 'None'
streamlineTubesDisplay.SelectNormalArray = 'TubeNormals'
streamlineTubesDisplay.SelectTangentArray = 'None'
streamlineTubesDisplay.OSPRayScaleArray = 'AngularVelocity'
streamlineTubesDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
streamlineTubesDisplay.SelectOrientationVectors = 'Normals'
streamlineTubesDisplay.ScaleFactor = 1.4016644001007081
streamlineTubesDisplay.SelectScaleArray = 'AngularVelocity'
streamlineTubesDisplay.GlyphType = 'Arrow'
streamlineTubesDisplay.GlyphTableIndexArray = 'AngularVelocity'
streamlineTubesDisplay.GaussianRadius = 0.0700832200050354
streamlineTubesDisplay.SetScaleArray = ['POINTS', 'AngularVelocity']
streamlineTubesDisplay.ScaleTransferFunction = 'PiecewiseFunction'
streamlineTubesDisplay.OpacityArray = ['POINTS', 'AngularVelocity']
streamlineTubesDisplay.OpacityTransferFunction = 'PiecewiseFunction'
streamlineTubesDisplay.DataAxesGrid = 'GridAxesRepresentation'
streamlineTubesDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
streamlineTubesDisplay.ScaleTransferFunction.Points = [-0.4927095146685523, 0.0, 0.5, 0.0, 0.6623530655842629, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
streamlineTubesDisplay.OpacityTransferFunction.Points = [-0.4927095146685523, 0.0, 0.5, 0.0, 0.6623530655842629, 1.0, 0.5, 0.0]

# trace defaults for the display properties.
streamlineArrowsDisplay.Representation = 'Surface'
streamlineArrowsDisplay.ColorArrayName = ['POINTS', 'velocities']
streamlineArrowsDisplay.LookupTable = velocitiesLUT
streamlineArrowsDisplay.SelectTCoordArray = 'None'
streamlineArrowsDisplay.SelectNormalArray = 'None'
streamlineArrowsDisplay.SelectTangentArray = 'None'
streamlineArrowsDisplay.OSPRayScaleArray = 'AngularVelocity'
streamlineArrowsDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
streamlineArrowsDisplay.SelectOrientationVectors = 'Normals'
streamlineArrowsDisplay.ScaleFactor = 1.0
streamlineArrowsDisplay.SelectScaleArray = 'AngularVelocity'
streamlineArrowsDisplay.GlyphType = 'Arrow'
streamlineArrowsDisplay.GlyphTableIndexArray = 'AngularVelocity'
streamlineArrowsDisplay.GaussianRadius = 1.0
streamlineArrowsDisplay.SetScaleArray = ['POINTS', 'AngularVelocity']
streamlineArrowsDisplay.ScaleTransferFunction = 'PiecewiseFunction'
streamlineArrowsDisplay.OpacityArray = ['POINTS', 'AngularVelocity']
streamlineArrowsDisplay.OpacityTransferFunction = 'PiecewiseFunction'
streamlineArrowsDisplay.DataAxesGrid = 'GridAxesRepresentation'
streamlineArrowsDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
streamlineArrowsDisplay.ScaleTransferFunction.Points = [-0.4594906762529114, 0.0, 0.5, 0.0, 0.5229818150748942, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
streamlineArrowsDisplay.OpacityTransferFunction.Points = [-0.4594906762529114, 0.0, 0.5, 0.0, 0.5229818150748942, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for velocitiesLUT in view renderView1
velocitiesLUTColorBar = GetScalarBar(velocitiesLUT, renderView1)
velocitiesLUTColorBar.Title = 'velocities'
velocitiesLUTColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
velocitiesLUTColorBar.Visibility = 0

Hide(fibers, renderView1)
fiberTubesDisplay = Show(fiberTubes, renderView1)
fiberTubesDisplay.AmbientColor = [0.4, 1.0, 0.57]
fiberTubesDisplay.ColorArrayName = ['POINTS', '']
fiberTubesDisplay.DiffuseColor = [0.4, 1.0, 0.57]


# hide data in view
Hide(streamlinePeripheryTubes, renderView1)

# hide data in view
Hide(streamlinePeripheryArrows, renderView1)

# hide data in view
Hide(streamlines, renderView1)

# hide data in view
Hide(streamlineTubes, renderView1)

# hide data in view
Hide(streamlineArrows, renderView1)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'velocities'
velocitiesPWF = GetOpacityTransferFunction('velocities')
velocitiesPWF.Points = [0.0040314331047620125, 0.0, 0.5, 0.0, 0.45681027798428525, 1.0, 0.5, 0.0]
velocitiesPWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# restore active source
SetActiveSource(streamlinePeripheryArrows)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')
