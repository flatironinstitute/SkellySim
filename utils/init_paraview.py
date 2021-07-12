import os
import toml
from pathlib import Path

sourcepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "paraview_utils")

def patch_script(script_name, toml_file):
    with  open(os.path.join(sourcepath, script_name), "r") as f:
        script = f.read()

    return 'toml_file = "' + str(os.path.realpath(toml_file)) + '"\n' + script


toml_files = sorted(Path('.').glob('*.toml'))
if 'toml_file' not in locals() and len(toml_files) > 1 or len(toml_files) == 0:
    print("Can't automatically determine configuration file, please set 'toml_file' variable to the path of the relevant configuration file.")
else:
    if 'toml_file' not in locals():
        toml_file = toml_files[0]

    with open(toml_file) as f:
        skelly_config = toml.load(f)


    if skelly_config['fibers']:
        print("Loading fiber source")

        fibers = ProgrammableSource(
            PythonPath="'{}'".format(sourcepath),
            Script=patch_script("fiber_reader.py", toml_file),
            ScriptRequestInformation=open(os.path.join(sourcepath, "fiber_reader_request.py"), "r").read(),
            OutputDataSetType="vtkPolyData",
            guiName="FiberReader",
        )
        print("Finished loading fiber source")

        tf = Tube(fibers)
        tf.Radius = 0.025
        tf.Capping = 1
        tf.NumberofSides = 10

        Show(tf)
        SetDisplayProperties(DiffuseColor=[0 / 255, 255 / 255, 127 / 255])

    if skelly_config['bodies']:
        print("Loading body source")
        bodies = ProgrammableSource(
            PythonPath="'{}'".format(sourcepath),
            Script=patch_script("body_reader.py", toml_file),
            ScriptRequestInformation=open(os.path.join(sourcepath, "body_reader_request.py"), "r").read(),
            OutputDataSetType="vtkMultiblockDataSet",
            guiName="BodyReader",
        )
        print("Finished loading body source")

        sf = Smooth(bodies)
        Show(sf)
        SetDisplayProperties(DiffuseColor=[143 / 255, 255 / 255, 246 / 255])

    if os.path.exists('skelly_sim.vf.0'):
        print("Loading velocity field")
        vf = ProgrammableSource(
            PythonPath="'{}'".format(sourcepath),
            Script=patch_script('field_reader.py', toml_file),
            ScriptRequestInformation=open(os.path.join(sourcepath, "field_reader_request.py"), "r").read(),
            OutputDataSetType="vtkPolyData",
            guiName="VelocityFieldReader",
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
