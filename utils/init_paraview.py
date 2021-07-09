import os

sourcepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "paraview_utils")

print("Loading fiber source")
fibers = ProgrammableSource(
    PythonPath="'{}'".format(sourcepath),
    Script=open(os.path.join(sourcepath, "fiber_reader.py"), "r").read(),
    ScriptRequestInformation=open(os.path.join(sourcepath, "fiber_reader_request.py"), "r").read(),
    OutputDataSetType="vtkPolyData",
    guiName="Fibers",
)
print("Finished loading fiber source")

tf = Tube(fibers)
tf.Radius = 0.025
tf.Capping = 1
tf.NumberofSides = 10

Show(tf)
SetDisplayProperties(DiffuseColor=[0 / 255, 255 / 255, 127 / 255])

print("Loading body source")
bodies = ProgrammableSource(
    PythonPath="'{}'".format(sourcepath),
    Script=open(os.path.join(sourcepath, "body_reader.py"), "r").read(),
    ScriptRequestInformation=open(os.path.join(sourcepath, "body_reader_request.py"), "r").read(),
    OutputDataSetType="vtkMultiblockDataSet",
    guiName="Bodies",
)
print("Finished loading body source")

sf = Smooth(bodies)
Show(sf)
SetDisplayProperties(DiffuseColor=[143 / 255, 255 / 255, 246 / 255])

if os.path.exists('skelly_sim.vf.0'):
    print("Loading velocity field")
    vf = ProgrammableSource(
        PythonPath="'{}'".format(sourcepath),
        Script=open(os.path.join(sourcepath, "field_reader.py"), "r").read(),
        ScriptRequestInformation=open(os.path.join(sourcepath, "field_reader_request.py"), "r").read(),
        OutputDataSetType="vtkPolyData",
        guiName="Velocity Field",
    )
    print("Finished loading velocity field source")

    glyph = Glyph(vf, guiName="Velocity Field Glyphs")
    glyph.ScaleFactor = 2.0
    glyph.OrientationArray = ('POINTS', 'velocities')
    display = Show(glyph)
    ColorBy(display, ('POINTS', 'velocities'))
    Hide(glyph)

    vi = PointVolumeInterpolator(Input=vf, Source='Bounded Volume', guiName="Volume Interpolator")

    st = StreamTracer(Input=vi, Vectors=('POINTS', 'velocities'), SeedType='Point Cloud', guiName="Stream Trace")
    st.SeedType.Radius = 6.0
    st.SeedType.Center = [0.0, 0.0, 0.0]
    st.MaximumStreamlineLength = 60.0

    sttube = Tube(st, guiName="StreamTracer Tube Filter")
    sttube.Radius = 0.05
    display = Show(sttube)
    ColorBy(display, ('POINTS', 'velocities'))

    stglyph = Glyph(st, guiName="StreamTracer Glyph Filter")
    stglyph.OrientationArray = ('POINTS', 'velocities')
    stglyph.ScaleArray = ('POINTS', 'No scale array')
    stglyph.ScaleFactor = 2.0
    stglyph.GlyphMode = 'Every Nth Point'
    stglyph.Stride = 20

    display = Show(stglyph)
    ColorBy(display, ('POINTS', 'velocities'))

    colorMap = GetColorTransferFunction('velocities')
    colorMap.VectorMode = 'Magnitude'
    scalarBar = GetScalarBar(colorMap)
    scalarBar.Title = 'Velocity'
    scalarBar.Visibility = 1

Show(AnnotateTime(guiName='Time'))
